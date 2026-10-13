#!/usr/bin/env bash
# Start, stop, or restart the RSI Web-UI. No arguments means restart.
# Pid and log stay under rsi/ so the repo root stays clean.
# macOS has no /proc: liveness uses kill -0, command lines fall back to ps.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RSI_DIR="${ROOT}/rsi"
PID_FILE="${RSI_DIR}/.rsi.pid"
LOG_FILE="${RSI_DIR}/.rsi.log"
LOCK_FILE="${RSI_DIR}/.rsi.lock"
PORT="${WEBUI_PORT:-${RSI_PORT:-7860}}"
HOST="${WEBUI_SERVER:-${RSI_HOST:-127.0.0.1}}"
PY=""
PORT_BLOCKED=0

browser_url() {
  local show="${HOST}"
  if [[ "${show}" == "0.0.0.0" || "${show}" == "::" || "${show}" == "[::]" ]]; then
    show="127.0.0.1"
  fi
  printf 'http://%s:%s/\n' "${show}" "${PORT}"
}

usage() {
  cat <<EOF
用法：./run.sh [start|stop|restart]
  start    后台启动 python -m rsi（默认 http://127.0.0.1:7860/）
  stop     结束本应用的进程，以及它占用的 ${PORT} 端口
  restart  先停止再启动
不带参数时等同于 restart。
日志：rsi/.rsi.log    进程号：rsi/.rsi.pid
EOF
}

pid_alive() {
  local pid="$1"
  [[ "${pid}" =~ ^[0-9]+$ ]] || return 1
  kill -0 "${pid}" 2>/dev/null
}

cmdline_of() {
  local pid="$1"
  if [[ -r "/proc/${pid}/cmdline" ]]; then
    tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true
    return 0
  fi
  ps -p "${pid}" -o command= 2>/dev/null || true
}

release_lock() {
  # { true >&9; } is the macOS-friendly way to see whether the lock fd is open.
  if { true >&9; } 2>/dev/null; then
    if command -v flock >/dev/null 2>&1; then
      flock -u 9 2>/dev/null || true
    fi
    exec 9>&-
  fi
}

resolve_python() {
  if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    PY="${VIRTUAL_ENV}/bin/python"
    echo "[依赖] 使用已激活的虚拟环境：${VIRTUAL_ENV}"
  elif [[ -x "${RSI_DIR}/.venv/bin/python" ]]; then
    PY="${RSI_DIR}/.venv/bin/python"
    echo "[依赖] 使用已有虚拟环境：rsi/.venv"
  elif [[ -x "${ROOT}/.venv/bin/python" ]]; then
    PY="${ROOT}/.venv/bin/python"
    echo "[依赖] 使用已有虚拟环境：.venv"
  elif command -v python3 >/dev/null 2>&1; then
    PY="$(command -v python3)"
    echo "[依赖] 使用 Python：${PY}"
  elif command -v python >/dev/null 2>&1; then
    PY="$(command -v python)"
    echo "[依赖] 使用 Python：${PY}"
  else
    echo "[依赖] 未找到 python3，无法启动。"
    exit 1
  fi
}

python_ready() {
  "${PY}" -c "import fastapi, yaml, uvicorn" >/dev/null 2>&1
}

install_requirements() {
  echo "[依赖] 未检测到 fastapi / uvicorn，正在执行 pip install -r rsi/requirements.txt …"
  if "${PY}" -m pip install -r "${RSI_DIR}/requirements.txt"; then
    return 0
  fi
  if [[ -n "${VIRTUAL_ENV:-}" || "${PY}" == "${RSI_DIR}/.venv/bin/python" || "${PY}" == "${ROOT}/.venv/bin/python" ]]; then
    echo "[依赖] 依赖安装失败。请查看上面的 pip 输出。"
    exit 1
  fi
  echo "[依赖] 当前解释器无法安装依赖，正在创建 rsi/.venv …"
  local base="${PY}"
  "${base}" -m venv "${RSI_DIR}/.venv"
  PY="${RSI_DIR}/.venv/bin/python"
  "${PY}" -m pip install -U pip
  "${PY}" -m pip install -r "${RSI_DIR}/requirements.txt"
}

ensure_deps() {
  resolve_python
  if python_ready; then
    echo "[依赖] 运行依赖已就绪。"
    return 0
  fi
  install_requirements
  if ! python_ready; then
    echo "[依赖] 安装后仍无法导入 fastapi / uvicorn。"
    exit 1
  fi
  echo "[依赖] 依赖安装完成。"
}

is_our_pid() {
  local pid="$1"
  local cmd recorded
  pid_alive "${pid}" || return 1
  cmd="$(cmdline_of "${pid}")"
  case "${cmd}" in
    *pytest*) return 1 ;;
  esac
  if [[ -f "${PID_FILE}" ]]; then
    recorded="$(tr -d '[:space:]' < "${PID_FILE}" || true)"
    if [[ "${recorded}" == "${pid}" && "${cmd}" == *python* ]]; then
      return 0
    fi
  fi
  case "${cmd}" in
    *"-m rsi"*|*"rsi/__main__.py"*|*"rsi/__main__ "*) return 0 ;;
  esac
  return 1
}

listeners_on_port() {
  if command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN -t 2>/dev/null || true
    return 0
  fi
  if command -v ss >/dev/null 2>&1; then
    ss -lptn "sport = :${PORT}" 2>/dev/null | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' || true
  fi
}

port_has_listener() {
  local pid
  for pid in $(listeners_on_port | sort -u); do
    [[ -n "${pid}" ]] && return 0
  done
  return 1
}

refresh_pid_from_listener() {
  local pid
  for pid in $(listeners_on_port | sort -u); do
    if is_our_pid "${pid}"; then
      echo "${pid}" > "${PID_FILE}"
      echo "${pid}"
      return 0
    fi
  done
  return 1
}

kill_tree() {
  local pid="$1"
  local child
  pid_alive "${pid}" || return 0
  if command -v pgrep >/dev/null 2>&1; then
    for child in $(pgrep -P "${pid}" 2>/dev/null || true); do
      kill_tree "${child}"
    done
  fi
  kill "${pid}" 2>/dev/null || true
}

wait_until_dead() {
  local pid="$1"
  local i
  for i in 1 2 3 4 5 6 7 8 9 10; do
    pid_alive "${pid}" || return 0
    sleep 0.3
  done
  kill -9 "${pid}" 2>/dev/null || true
  sleep 0.2
}

cmd_stop() {
  local pid recorded foreign="" stopped=0
  PORT_BLOCKED=0
  echo "[停止] 正在停止 RSI Web-UI …"
  if [[ -f "${PID_FILE}" ]]; then
    recorded="$(tr -d '[:space:]' < "${PID_FILE}" || true)"
    if [[ -n "${recorded}" ]] && is_our_pid "${recorded}"; then
      echo "[停止] 结束进程 ${recorded}。"
      kill_tree "${recorded}"
      wait_until_dead "${recorded}"
      stopped=1
    fi
  fi
  for pid in $(listeners_on_port | sort -u); do
    if is_our_pid "${pid}"; then
      echo "[停止] 结束占用端口 ${PORT} 的本应用进程 ${pid}。"
      kill_tree "${pid}"
      wait_until_dead "${pid}"
      stopped=1
    else
      foreign="${foreign} ${pid}"
    fi
  done
  rm -f "${PID_FILE}"
  if [[ -n "${foreign}" ]]; then
    PORT_BLOCKED=1
    echo "[停止] 端口 ${PORT} 仍被其他程序占用（pid${foreign}），未结束该进程。"
    return 0
  fi
  if [[ "${stopped}" == "1" ]]; then
    echo "[停止] 已停止。地址 $(browser_url) 不再由本应用提供。"
  else
    echo "[停止] RSI Web-UI 未在运行。"
  fi
}

wait_until_ready() {
  local pid="$1"
  local i url
  url="$(browser_url)"
  for i in $(seq 1 90); do
    if curl -fsS -o /dev/null --max-time 2 "${url}" 2>/dev/null; then
      refresh_pid_from_listener >/dev/null 2>&1 || true
      return 0
    fi
    # A stale $! can die while the real listener is still coming up. Fail only
    # when the recorded pid is dead and nothing is listening.
    if ! pid_alive "${pid}" && ! port_has_listener; then
      echo "[启动] 进程已退出，且端口 ${PORT} 上没有服务。最近日志："
      tail -n 40 "${LOG_FILE}" 2>/dev/null || true
      rm -f "${PID_FILE}"
      return 1
    fi
    sleep 1
  done
  if port_has_listener; then
    refresh_pid_from_listener >/dev/null 2>&1 || true
    return 0
  fi
  echo "[启动] 90 秒内未打开 ${url}。最近日志："
  tail -n 40 "${LOG_FILE}" 2>/dev/null || true
  return 1
}

cmd_start() {
  local pid existing url
  url="$(browser_url)"
  if [[ "${PORT_BLOCKED}" == "1" ]]; then
    echo "[启动] 端口 ${PORT} 被其他程序占用，未启动。可设置 WEBUI_PORT 换一个端口。"
    exit 1
  fi
  if [[ -f "${PID_FILE}" ]]; then
    existing="$(tr -d '[:space:]' < "${PID_FILE}" || true)"
    if [[ -n "${existing}" ]] && is_our_pid "${existing}"; then
      echo "[启动] RSI Web-UI 已在运行：${url}"
      echo "[启动] 进程 ${existing}，日志 rsi/.rsi.log"
      return 0
    fi
    rm -f "${PID_FILE}"
  fi
  for pid in $(listeners_on_port | sort -u); do
    if is_our_pid "${pid}"; then
      echo "${pid}" > "${PID_FILE}"
      echo "[启动] RSI Web-UI 已在运行：${url}"
      echo "[启动] 进程 ${pid}，日志 rsi/.rsi.log"
      return 0
    fi
    echo "[启动] 端口 ${PORT} 已被其他程序占用（pid ${pid}），未启动。可设置 WEBUI_PORT 换一个端口。"
    exit 1
  done
  ensure_deps
  echo "[启动] 正在后台启动 python -m rsi …"
  {
    echo ""
    echo "===== $(date '+%Y-%m-%d %H:%M:%S') start port=${PORT} ====="
  } >> "${LOG_FILE}"
  cd "${ROOT}"
  PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    PYTHONUNBUFFERED=1 \
    WEBUI_PORT="${PORT}" \
    WEBUI_SERVER="${HOST}" \
    nohup "${PY}" -m rsi >> "${LOG_FILE}" 2>&1 9>&- </dev/null &
  pid="$!"
  disown "${pid}" 2>/dev/null || true
  echo "${pid}" > "${PID_FILE}"
  release_lock
  if ! wait_until_ready "${pid}"; then
    echo "[启动] 启动失败。请查看 rsi/.rsi.log"
    exit 1
  fi
  if [[ -f "${PID_FILE}" ]]; then
    pid="$(tr -d '[:space:]' < "${PID_FILE}")"
  fi
  echo "[启动] 已启动：${url}"
  echo "[启动] 进程 ${pid}，日志 rsi/.rsi.log"
}

cmd_restart() {
  echo "[重启] 先停止，再启动 RSI Web-UI。"
  cmd_stop
  cmd_start
}

main() {
  local action="${1:-restart}"
  if [[ $# -gt 1 ]]; then
    usage
    exit 2
  fi
  mkdir -p "${RSI_DIR}"
  if command -v flock >/dev/null 2>&1; then
    exec 9>"${LOCK_FILE}"
    if ! flock -n 9; then
      echo "[错误] 另一个 ./run.sh 正在执行，请稍后再试。"
      exit 1
    fi
  fi
  case "${action}" in
    start) cmd_start ;;
    stop) cmd_stop ;;
    restart) cmd_restart ;;
    -h|--help|help) usage ;;
    *)
      usage
      exit 2
      ;;
  esac
}

main "$@"

#!/usr/bin/env bash
# Sync cursor/rsi-diag-card-ui-20261019-4ed3 between GitHub (origin) and Meituan (sankuai).
# Cherry-picks sankuai-only commits onto the GitHub branch, pushes GitHub, then sankuai.
# A non-fast-forward sankuai update stops unless --force-sankuai is passed.
set -euo pipefail

BRANCH="cursor/rsi-diag-card-ui-20261019-4ed3"
ORIGIN_URL="git@github.com:achievec/ms-swift.git"
SANKUAI_URL="ssh://git@git.sankuai.com/~liyuhao03/ms-swift.git"
FORCE_SANKUAI=0

usage() {
  cat <<EOF
用法：rsi/scripts/sync-to-sankuai.sh [--force-sankuai]
  在仓库根目录同步分支 ${BRANCH}：
  先从 GitHub (origin) 快进拉取，再把 sankuai 上多出来的提交按顺序 cherry-pick 进来，
  推到 GitHub，然后推到 sankuai。
  工作区有未提交改动时会直接退出。
  cherry-pick 会生成新哈希，推 sankuai 可能不是快进。默认不强推。
  --force-sankuai  仅对 sankuai 使用 --force-with-lease
EOF
}

die() {
  echo "$1" >&2
  exit 1
}

for arg in "$@"; do
  case "${arg}" in
    --force-sankuai) FORCE_SANKUAI=1 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "未知参数：${arg}" >&2
      usage >&2
      exit 2
      ;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ "${ROOT}" != "/" && ! -e "${ROOT}/.git" ]]; do
  ROOT="$(dirname "${ROOT}")"
done
[[ -e "${ROOT}/.git" ]] || die "没有在脚本上级目录找到 git 仓库。"
cd "${ROOT}"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "工作区有未提交改动，已停止。先处理完再同步。" >&2
  git status -sb >&2
  exit 1
fi

if ! git remote get-url origin >/dev/null 2>&1; then
  echo "未找到 origin，正在添加 ${ORIGIN_URL}"
  git remote add origin "${ORIGIN_URL}"
fi
if ! git remote get-url sankuai >/dev/null 2>&1; then
  echo "未找到 sankuai，正在添加 ${SANKUAI_URL}"
  git remote add sankuai "${SANKUAI_URL}"
fi

echo "从 GitHub 拉取 ${BRANCH} …"
if ! git fetch origin "${BRANCH}"; then
  die "无法从 GitHub (origin) 拉取 ${BRANCH}。"
fi
if git show-ref --verify --quiet "refs/heads/${BRANCH}"; then
  git checkout "${BRANCH}"
else
  git checkout -b "${BRANCH}" "origin/${BRANCH}"
fi
if ! git merge --ff-only "origin/${BRANCH}"; then
  die "无法快进到 origin/${BRANCH}。本地和 GitHub 历史不一致，已停止。请先手动处理。"
fi

sankuai_line=""
if ! sankuai_line="$(git ls-remote --heads sankuai "refs/heads/${BRANCH}")"; then
  die "无法连接 sankuai（${SANKUAI_URL}）。"
fi
if [[ -n "${sankuai_line}" ]]; then
  git fetch sankuai "${BRANCH}"
  to_pick=()
  while IFS= read -r sha; do
    [[ -n "${sha}" ]] || continue
    mark="$(git cherry HEAD "sankuai/${BRANCH}" | awk -v s="${sha}" '$2 == s { print $1; exit }')"
    if [[ "${mark}" == "-" ]]; then
      continue
    fi
    to_pick+=("${sha}")
  done < <(git rev-list --reverse "HEAD..sankuai/${BRANCH}")
  if [[ "${#to_pick[@]}" -gt 0 ]]; then
    echo "sankuai 上有 ${#to_pick[@]} 个提交不在当前分支，按从旧到新 cherry-pick。"
    for sha in "${to_pick[@]}"; do
      echo "cherry-pick $(git log -1 --oneline "${sha}")"
      if ! git cherry-pick "${sha}"; then
        git cherry-pick --abort >/dev/null 2>&1 || true
        die "cherry-pick 失败，已中止并尽量恢复工作区。请手动处理 ${sha} 后再运行本脚本。"
      fi
    done
  else
    echo "sankuai 上没有需要 cherry-pick 的新提交。"
  fi
else
  echo "sankuai 上还没有 ${BRANCH}，稍后会新建。"
fi

echo "推送到 GitHub (origin) …"
if ! git push origin "${BRANCH}"; then
  die "推送到 GitHub 失败。没有推 sankuai。请先处理 origin 后再运行。"
fi

echo "推送到 sankuai …"
push_out=""
push_status=0
push_out="$(git push sankuai "${BRANCH}" 2>&1)" || push_status=$?
if [[ -n "${push_out}" ]]; then
  printf '%s\n' "${push_out}"
fi
if [[ "${push_status}" -ne 0 ]]; then
  non_ff=0
  case "${push_out}" in
    *non-fast-forward*|*"[rejected]"*|*"Updates were rejected"*) non_ff=1 ;;
  esac
  if [[ "${non_ff}" -ne 1 ]]; then
    die "推送到 sankuai 失败（不是快进拒绝）。没有强制覆盖。"
  fi
  if [[ "${FORCE_SANKUAI}" -ne 1 ]]; then
    cat >&2 <<EOF
推送到 sankuai 不是快进，已停止，没有强制覆盖。

sankuai 上还有当前分支不包括的旧提交。cherry-pick 会写成新的提交哈希，所以普通 git push 会被拒绝。
这些提交已经推到 GitHub。若确认用现在这条历史覆盖 sankuai 上的 ${BRANCH}，请再执行：

  rsi/scripts/sync-to-sankuai.sh --force-sankuai

该参数只对 sankuai 使用 --force-with-lease，不会强制推送 GitHub。
若 sankuai 在本次拉取之后又有新提交，force-with-lease 会失败，需要重新同步。
EOF
    exit 1
  fi
  echo "按 --force-sankuai，对 sankuai 使用 --force-with-lease。"
  if ! git push --force-with-lease="${BRANCH}:sankuai/${BRANCH}" sankuai "${BRANCH}"; then
    die "force-with-lease 失败。sankuai 上的 ${BRANCH} 在拉取之后变过，已停止。"
  fi
fi

echo "同步完成。"
git status -sb
git log -1 --oneline

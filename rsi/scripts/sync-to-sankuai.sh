#!/usr/bin/env bash
# Sync cursor/rsi-diag-card-ui-20261019-4ed3 between GitHub and Meituan (sankuai).
# Remotes are resolved by URL (github.com / git.sankuai.com), not by the name "origin",
# so both layouts work: origin=GitHub or origin=sankuai + github=GitHub.
# Cherry-picks sankuai-only commits onto the GitHub tip, pushes GitHub, then sankuai.
# A non-fast-forward sankuai update stops unless --force-sankuai is passed.
set -euo pipefail

BRANCH="cursor/rsi-diag-card-ui-20261019-4ed3"
GITHUB_URL="git@github.com:achievec/ms-swift.git"
SANKUAI_URL="ssh://git@git.sankuai.com/~liyuhao03/ms-swift.git"
FORCE_SANKUAI=0
GITHUB_ONLY=0
SANKUAI_ONLY=0

usage() {
  cat <<EOF
用法：rsi/scripts/sync-to-sankuai.sh [--force-sankuai] [--github-only|--sankuai-only]
  在仓库根目录同步分支 ${BRANCH}：
  按远程 URL 识别 GitHub（含 github.com）与 sankuai（含 git.sankuai.com），
  不依赖「origin 一定是 GitHub」。常见布局：
    - origin → GitHub，另有 sankuai；或
    - origin / sankuai → 美团仓，另有 github → GitHub
  默认：从 GitHub 快进拉取 → 把 sankuai 多出来的提交 cherry-pick 进来 →
  推 GitHub → 推 sankuai。
  工作区有未提交改动时会直接退出。
  cherry-pick 会生成新哈希，推 sankuai 可能不是快进。默认不强推。
  --force-sankuai  仅对 sankuai 使用 --force-with-lease
  --github-only    只拉取并推送到 GitHub（不碰 sankuai）
  --sankuai-only   只推送到 sankuai（本地须已在 ${BRANCH}；可与 --force-sankuai 合用）
EOF
}

die() {
  echo "$1" >&2
  exit 1
}

for arg in "$@"; do
  case "${arg}" in
    --force-sankuai) FORCE_SANKUAI=1 ;;
    --github-only) GITHUB_ONLY=1 ;;
    --sankuai-only) SANKUAI_ONLY=1 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "未知参数：${arg}" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "${GITHUB_ONLY}" -eq 1 && "${SANKUAI_ONLY}" -eq 1 ]]; then
  die "不能同时使用 --github-only 与 --sankuai-only。"
fi

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

# Prefer an existing remote whose URL matches; otherwise add under the given name.
# If several remotes match (e.g. origin + sankuai both point at Meituan), prefer prefer_name.
ensure_remote() {
  local prefer_name="$1"
  local url="$2"
  local needle="$3"
  local name url_now fallback=""
  while IFS=$'\t' read -r name url_now; do
    [[ "${url_now}" == *"${needle}"* ]] || continue
    if [[ "${name}" == "${prefer_name}" ]]; then
      printf '%s' "${name}"
      return 0
    fi
    [[ -z "${fallback}" ]] && fallback="${name}"
  done < <(git remote -v | awk '/\(fetch\)$/ { print $1 "\t" $2 }')
  if [[ -n "${fallback}" ]]; then
    printf '%s' "${fallback}"
    return 0
  fi
  if git remote get-url "${prefer_name}" >/dev/null 2>&1; then
    die "远程 ${prefer_name} 已存在，但 URL 不是 ${needle}（当前：$(git remote get-url "${prefer_name}")）。请改名或改 URL 后再跑。"
  fi
  echo "未找到含 ${needle} 的远程，正在添加 ${prefer_name} → ${url}" >&2
  git remote add "${prefer_name}" "${url}"
  printf '%s' "${prefer_name}"
}

GITHUB_REMOTE="$(ensure_remote github "${GITHUB_URL}" "github.com")"
SANKUAI_REMOTE=""
if [[ "${GITHUB_ONLY}" -ne 1 ]]; then
  SANKUAI_REMOTE="$(ensure_remote sankuai "${SANKUAI_URL}" "git.sankuai.com")"
fi

echo "远程：GitHub=${GITHUB_REMOTE}  sankuai=${SANKUAI_REMOTE:-（跳过）}"

checkout_branch() {
  if git show-ref --verify --quiet "refs/heads/${BRANCH}"; then
    git checkout "${BRANCH}"
  else
    die "本地没有分支 ${BRANCH}。请先创建或从某一远程拉取。"
  fi
}

push_sankuai() {
  echo "推送到 sankuai (${SANKUAI_REMOTE}) …"
  local push_out="" push_status=0 non_ff=0
  push_out="$(git push "${SANKUAI_REMOTE}" "${BRANCH}" 2>&1)" || push_status=$?
  if [[ -n "${push_out}" ]]; then
    printf '%s\n' "${push_out}"
  fi
  if [[ "${push_status}" -eq 0 ]]; then
    return 0
  fi
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
若确认用现在这条历史覆盖 sankuai 上的 ${BRANCH}，请再执行：

  rsi/scripts/sync-to-sankuai.sh --force-sankuai
  # 或仅推 sankuai：
  rsi/scripts/sync-to-sankuai.sh --sankuai-only --force-sankuai

该参数只对 sankuai 使用 --force-with-lease，不会强制推送 GitHub。
若 sankuai 在本次拉取之后又有新提交，force-with-lease 会失败，需要重新同步。
EOF
    exit 1
  fi
  echo "按 --force-sankuai，对 sankuai 使用 --force-with-lease。"
  if ! git push --force-with-lease="${BRANCH}:${SANKUAI_REMOTE}/${BRANCH}" "${SANKUAI_REMOTE}" "${BRANCH}"; then
    die "force-with-lease 失败。sankuai 上的 ${BRANCH} 在拉取之后变过，已停止。"
  fi
}

if [[ "${SANKUAI_ONLY}" -eq 1 ]]; then
  checkout_branch
  echo "按 --sankuai-only，跳过 GitHub 拉取与推送。"
  git fetch "${SANKUAI_REMOTE}" "${BRANCH}" >/dev/null 2>&1 || true
  push_sankuai
  echo "同步完成（仅 sankuai）。"
  git status -sb
  git log -1 --oneline
  exit 0
fi

echo "从 GitHub (${GITHUB_REMOTE}) 拉取 ${BRANCH} …"
if ! git fetch "${GITHUB_REMOTE}" "${BRANCH}"; then
  # Branch may not exist on GitHub yet — allow first push.
  if ! git ls-remote --heads "${GITHUB_REMOTE}" "refs/heads/${BRANCH}" | grep -q .; then
    echo "GitHub 上还没有 ${BRANCH}，将用本地分支首次推送。"
    checkout_branch
  else
    die "无法从 GitHub (${GITHUB_REMOTE}) 拉取 ${BRANCH}。"
  fi
else
  if git show-ref --verify --quiet "refs/heads/${BRANCH}"; then
    git checkout "${BRANCH}"
  else
    git checkout -b "${BRANCH}" "${GITHUB_REMOTE}/${BRANCH}"
  fi
  if git show-ref --verify --quiet "refs/remotes/${GITHUB_REMOTE}/${BRANCH}"; then
    if ! git merge --ff-only "${GITHUB_REMOTE}/${BRANCH}"; then
      die "无法快进到 ${GITHUB_REMOTE}/${BRANCH}。本地和 GitHub 历史不一致，已停止。请先手动处理。"
    fi
  fi
fi

if [[ "${GITHUB_ONLY}" -ne 1 ]]; then
  sankuai_line=""
  if ! sankuai_line="$(git ls-remote --heads "${SANKUAI_REMOTE}" "refs/heads/${BRANCH}")"; then
    die "无法连接 sankuai（${SANKUAI_URL}）。"
  fi
  if [[ -n "${sankuai_line}" ]]; then
    git fetch "${SANKUAI_REMOTE}" "${BRANCH}"
    to_pick=()
    while IFS= read -r sha; do
      [[ -n "${sha}" ]] || continue
      mark="$(git cherry HEAD "${SANKUAI_REMOTE}/${BRANCH}" | awk -v s="${sha}" '$2 == s { print $1; exit }')"
      if [[ "${mark}" == "-" ]]; then
        continue
      fi
      to_pick+=("${sha}")
    done < <(git rev-list --reverse "HEAD..${SANKUAI_REMOTE}/${BRANCH}")
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
fi

echo "推送到 GitHub (${GITHUB_REMOTE}) …"
if ! git push -u "${GITHUB_REMOTE}" "${BRANCH}"; then
  die "推送到 GitHub 失败。没有推 sankuai。请先处理 GitHub 后再运行。"
fi

if [[ "${GITHUB_ONLY}" -eq 1 ]]; then
  echo "同步完成（仅 GitHub）。"
  git status -sb
  git log -1 --oneline
  exit 0
fi

push_sankuai

echo "同步完成。"
git status -sb
git log -1 --oneline

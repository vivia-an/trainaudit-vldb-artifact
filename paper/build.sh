#!/bin/bash
# 用法:
#   ./build.sh            # 默认编译 main
#   ./build.sh cn         # 只编译 main_cn (中文版用 xelatex)
#   ./build.sh all        # 同时编译 main 和 main_cn
set -e
cd "$(dirname "$0")"

# 本容器无系统 texlive，按集群依次探测共享 TeXLive
for TEXLIVE_BIN in \
    "/volume/pt-train/users/wangqing/tools/2025/texlive/2025/bin/x86_64-linux" \
    "/volume/qscai/cqs/workspace/paper/texlive/bin/x86_64-linux"; do
    if [ -d "$TEXLIVE_BIN" ]; then
        export PATH="$TEXLIVE_BIN:$PATH"
        break
    fi
done

# open 是 macOS 命令，无 GUI 环境跳过而不报错
open() { command open "$@" 2>/dev/null || true; }

build_one() {
    local doc=$1
    local engine=$2
    "$engine" -synctex=1 -shell-escape -interaction=nonstopmode "$doc.tex"
    bibtex "$doc"
    "$engine" -synctex=1 -shell-escape -interaction=nonstopmode "$doc.tex"
    "$engine" -synctex=1 -shell-escape -interaction=nonstopmode "$doc.tex"
}

case "${1:-main}" in
    main) build_one main pdflatex; open main.pdf ;;
    cn)   build_one main_cn xelatex; open main_cn.pdf ;;
    all)  build_one main pdflatex; build_one main_cn xelatex; open main.pdf ;;
    *) echo "unknown target: $1 (use main | cn | all)"; exit 1 ;;
esac

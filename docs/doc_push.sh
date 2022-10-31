#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# Builds docs from the checkedout HEAD
# and pushes the artifacts to gh-pages branch in github.com/pytorch/multipy
#
# 1. sphinx generated docs are copied to <repo-root>/<version>
# 2. if a release tag is found on HEAD then redirects are copied to <repo-root>/latest
# 3. if no release tag is found on HEAD then redirects are copied to <repo-root>/main
#
# gh-pages branch should look as follows:
# <repo-root>
#           |- 0.1.0rc2
#           |- 0.1.0rc3
#           |- <versions...>
#           |- main (redirects to the most recent ver in trunk, including release)
#           |- latest (redirects to the most recent release)
# If the most recent  release is 0.1.0 and main is at 0.1.1rc1 then,
# https://pytorch.org/multipy/main -> https://pytorch.org/multipy/0.1.1rc1
# https://pytorch.org/multipy/latest -> https://pytorch.org/multipy/0.1.0
#
# Redirects are done via Jekyll redirect-from  plugin. See:
#   sources/scripts/create_redirect_md.py
#   Makefile (redirect target)
#  (on gh-pages branch) _layouts/docs_redirect.html

set -ex

dry_run=0
for arg in "$@"; do
    shift
    case "$arg" in
        "--dry-run") dry_run=1 ;;
        "--help") echo "Usage $0 [--dry-run]"; exit 0 ;;
    esac
done

repo_origin="$(git remote get-url origin)"
repo_root=$(git rev-parse --show-toplevel)
branch=$(git rev-parse --abbrev-ref HEAD)
commit_id=$(git rev-parse --short HEAD)

if ! release_tag=$(git describe --tags --exact-match HEAD 2>/dev/null); then
    echo "No release tag found, building docs for main..."
    redirects=(main)
    release_tag="main"
else
    echo "Release tag $release_tag found, building docs for release..."
    redirects=(latest main)
fi

echo "Installing multipy from $repo_root..."
cd "$repo_root" || exit

# Here we hardcode versions until we
# find a better way to do it.

# multipy_ver="0.1.0dev0"
 multipy_ver="latest"

echo "Building multipy-$multipy_ver docs..."
docs_dir=$repo_root/docs
build_dir=$docs_dir/build
cd "$docs_dir" || exit
python3 -m pip install setuptools
python3 -m pip install -r requirements.txt
make clean html
echo "Doc build complete"

tmp_dir=/tmp/multipy_docs_tmp
rm -rf "${tmp_dir:?}"

echo "Checking out gh-pages branch..."
gh_pages_dir="$tmp_dir/multipy_gh_pages"
git clone -b gh-pages --single-branch "$repo_origin"  $gh_pages_dir

echo "Copying doc pages for $multipy_ver into $gh_pages_dir..."
rm -rf "${gh_pages_dir:?}/${multipy_ver:?}"
cp -R "$build_dir/html" "$gh_pages_dir/$multipy_ver"

cd $gh_pages_dir || exit

for redirect in "${redirects[@]}"; do
  echo "Creating redirect symlinks for: $redirect -> $multipy_ver..."
  rm -rf "${gh_pages_dir:?}/${redirect:?}"
  ln -s "$multipy_ver" "$redirect"
done

git add .
git commit --quiet -m "[doc_push][$release_tag] built from $commit_id ($branch). Redirects: ${redirects[*]} -> $multipy_ver."

if [ $dry_run -eq 1 ]; then
    echo "*** --dry-run mode, skipping push to gh-pages branch. To publish run: cd ${gh_pages_dir} && git push"
    exit
fi

git push

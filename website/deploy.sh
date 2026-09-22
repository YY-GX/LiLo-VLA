#!/usr/bin/env bash
# Build the project page and stage the result at the branch root, which is what
# GitHub Pages serves at https://yy-gx.github.io/LiLo-VLA/
#
#   cd website && ./deploy.sh && cd .. && git add -A && git commit && git push
set -euo pipefail

cd "$(dirname "$0")"
npm install --no-audit --no-fund
npm run build

cd ..
# Clear the previously built files, but never the source or git metadata.
find . -maxdepth 1 -mindepth 1 \
  -not -name .git -not -name website -not -name .nojekyll \
  -exec rm -rf {} +

cp -R website/dist/. .
touch .nojekyll
echo "Built site staged at the branch root. Commit and push to publish."

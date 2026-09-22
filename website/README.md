# LiLo-VLA project page

Source for <https://yy-gx.github.io/LiLo-VLA/>. Everything to do with the website
lives on this `gh-pages` branch: the built site sits at the branch root, which
is what GitHub Pages serves, and its source sits here in `website/`.

Built with [Astro](https://astro.build/) and Tailwind, from
[RomanHauksson/academic-project-astro-template](https://github.com/RomanHauksson/academic-project-astro-template),
the same template as the [EGR](https://yy-gx.github.io/EGR/) and
[HALTER](https://yy-gx.github.io/HALTER/) pages.

## Editing

```bash
cd website
npm install
npm run dev      # local preview at http://localhost:4321/LiLo-VLA/
```

Page content is a single file: `src/paper.mdx`. Components live in
`src/components/`, the colour and type tokens in `src/styles/global.css`.

The in-page table of contents is declared twice and the two must agree:
`<Contents items={…}>` in `src/paper.mdx`, and `navItems` in
`src/pages/index.astro`, which feeds the sticky `PageNav`.

## Assets

- `src/assets/` — figures that go through Astro's image optimizer.
- `public/static/` — served verbatim, at the **legacy paths** of the previous
  Bulma page. `public/static/pdfs/main.pdf` is what keeps
  <https://yy-gx.github.io/LiLo-VLA/static/pdfs/main.pdf> resolving; the code
  release links that URL. Do not move it.
- `public/videos/` — page media that has no legacy path (the animated
  architecture figure).

## Publishing

```bash
cd website
./deploy.sh                  # builds and stages the site at the branch root
cd ..
git add -A && git commit -m "Update project page" && git push
```

`deploy.sh` clears the previously built files from the branch root, leaving
`website/`, `.git` and `.nojekyll` alone, then copies the fresh build over.
Anything that must stay reachable therefore has to live in `website/public/`.

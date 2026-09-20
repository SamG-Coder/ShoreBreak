# Publishing ShoreBreak

The public source repository is
[cryptomanavan/ShoreBreak](https://github.com/cryptomanavan/ShoreBreak).
The [live demo](https://shorebreak-living-coast.netlify.app/) is a separate
Netlify deployment; cloning or forking this repository does not reconfigure it.

## Contribute or publish your own fork

Fork the repository on GitHub, then clone your fork and follow the
[development guide](DEVELOPMENT.md). Open a pull request to contribute changes
to ShoreBreak. Keep the project copyright and third-party notices when
redistributing source or built output.

The code is MIT licensed; texture and dependency exceptions are listed in
[asset credits](../ASSETS.md) and [third-party notices](../THIRD_PARTY_NOTICES.md).
Check the [Actions page](https://github.com/cryptomanavan/ShoreBreak/actions)
for Linux and Windows build results. Authenticate Git through your normal
credential manager or CLI; never put access tokens in source files or URLs.

## Build a Netlify deployment

In Netlify, import your repository with these settings:

| Setting | Value |
| --- | --- |
| Production branch | `main` |
| Base directory | Repository root |
| Build command | `npm run build` |
| Publish directory | `dist` |
| Node.js | 24 |
| Required environment variables | None |

The included `netlify.toml` supplies the build defaults. For manual deployment,
run `npm ci` and `npm run build`, then deploy the **contents of `dist/`** using
your own Netlify account. Publish at a domain/subdomain root, not a subpath.

Keep `THIRD-PARTY-NOTICES.txt` in the deployed output; Vite copies it from `public/`.
For changes to assets under immutable cached paths, use a new versioned asset
directory and update the source references.

## Create a source release ZIP

After committing a release:

```sh
git tag v0.1.0
git archive --format=zip --prefix=shorebreak/ --output=../shorebreak-open-source-v0.1.0.zip v0.1.0
```

Push the tag with `git push origin v0.1.0`, then create a GitHub release for that
tag. The source ZIP intentionally excludes installed dependencies, `dist/`,
local recordings, deploy credentials, and `.git` history. Recipients build with
`npm ci` and `npm run build`.

# hql-extensions

Official packages for [HQL](https://github.com/ewiger/hql), and the registry
of known ones. A package is a Cargo crate whose library is an HQL extension:
the implementations of the external declarations of the modules it provides,
beside those modules' contracts. `hql package install <name>` builds a
package from this repository with your `cargo` against the `hql` you run and
installs it into the project, `.hql/packages/<name>/`, or with `--home` into
the HQL home, `~/.hql/packages/<name>/`; `import <module>` in a program then
reaches it as it reaches a standard module. The design is
[HQL-0057](https://github.com/ewiger/hql/blob/main/doc/proposals/HQL-0057/index.hmd),
and the two levels are
[HQL-0059](https://github.com/ewiger/hql/blob/main/doc/proposals/HQL-0059/index.hmd).

| Package | Crate | Provides | What it does |
| --- | --- | --- | --- |
| [`ml`](crates/hql-ext-ml/) | `hql-ext-ml` | `ml`, `ml.metrics` | datasets from records, linear and logistic regression and k-means as values, prediction, and metrics |

## The registry

[`hql-packages.yaml`](hql-packages.yaml) is the registry `hql package` reads.
It lists every crate under `crates/` (an **official** package) and every
crate kept elsewhere that is admitted by name (a **known** package).
Installation by name reaches these entries and nothing else; a package on
your machine installs with `hql package install --path <dir>`.

To register a crate of your own, open a pull request that adds one `known`
entry. The review checks that:

- the crate's name begins with `hql-ext-`, its package name is one lowercase
  word that no builtin or registered package uses, and it provides modules
  under that name only;
- `[package.metadata.hql]` in its `Cargo.toml` states the package name, the
  modules it provides and its module root, and the contracts under that root
  declare exactly those modules;
- the entry's source is pinned, a crates.io version or a git `rev`, and its
  `hql` requirement names the versions it builds against;
- the crate builds and its `examples/` pass against the `hql` versions it
  claims, which CI runs.

An extension is native code with the rights of the `hql` process. The list
is reviewed for that reason, and an entry is removed when its source stops
building or its author asks.

## Developing a package

```sh
git clone https://github.com/ewiger/hql-extensions && cd hql-extensions
cargo test -p hql-ext-ml
hql package install --home --path crates/hql-ext-ml
hql eval 'import ml
Dataset([[1.0f], [2.0f], [3.0f]], target = [3.0f, 5.0f, 7.0f]).fit_linear()'
```

While a crate is being changed, a project's `.hql/config.toml` names it
instead of installing, and `hql` builds it on first use and again when its
source is newer than the build. The path is read against the project root,
the directory holding `.hql/`:

```toml
[packages]
paths = ["../hql-extensions/crates/hql-ext-ml"]
```

Each crate's `examples/` is such a project: `cd crates/hql-ext-ml/examples
&& hql run linear.hql` runs an example against the checkout.
`sh scripts/check-examples.sh` runs every example and holds it to its header,
which is what CI does.

A package is built against the `hql` that installs it, with that `hql`'s
compiler. An `hql` built from a checkout builds packages against that
checkout; a released `hql` builds them against the released crate of its own
version. `HQL_SOURCE=<dir>` names a checkout explicitly.

## Releasing a package

Each crate is published to crates.io on its own under its crate name. To
release one, set its `version` in `Cargo.toml` and in its registry entry,
merge to `main`, and push the tag `<crate>-v<version>`:

```sh
git tag hql-ext-ml-v0.1.0 && git push origin hql-ext-ml-v0.1.0
```

[`release.yml`](.github/workflows/release.yml) checks that the tag, the
crate's `Cargo.toml` and its registry entry agree, then runs `cargo publish`
for that crate only. A crate depends on `hql` by a version range, so the
`hql` version it names is on crates.io before the crate is tagged. It authenticates by crates.io trusted publishing, which
each crate's crates.io settings enable for this repository, the workflow
`release.yml` and the environment `crates-io`. A crate's first version
predates those settings: publish it with the repository secret
`CARGO_REGISTRY_TOKEN` set, then enable trusted publishing and remove the
secret.

## State

`hql` 0.7.0 carries `hql::ext`, the loader and `hql package`. `ml` installs
from a checkout with `--path` now, and by name once `hql` 0.7.0 is on
crates.io and the tag `hql-ext-ml-v0.1.0` is pushed. `Cargo.lock` holds
`hql` as resolved against a checkout until then; run `cargo update -p hql`
once 0.7.0 is published, before the first tag.

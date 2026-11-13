# hql-extensions

Official packages for [HQL](https://github.com/ewiger/hql), and the registry
of known ones. A package is a Cargo crate whose library is an HQL extension:
the implementations of the external declarations of the modules it provides,
beside those modules' contracts. `hql package install <name>` builds a
package from this repository with your `cargo` against the `hql` you run and
installs it into `~/.hql/lib/<name>/`; `import <module>` in a program then
reaches it as it reaches a standard module. The design is
[HQL-0057](https://github.com/ewiger/hql/blob/main/doc/proposals/HQL-0057/README.md).

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
cargo build -p hql-ext-ml
hql package install --path crates/hql-ext-ml
hql eval 'import ml  [[1.0, 2.0], [2.0, 4.0]].Dataset(target = [3.0, 6.0]).fit_linear()'
```

While a crate is being changed, a workspace's `hql.toml` names it instead of
installing:

```toml
[packages]
paths = ["../hql-extensions/crates/hql-ext-ml"]
```

## State

The mechanism is being built on hql's `feat/extensions-ml` branch (issue
0077). `hql` has no `package` command yet and no `hql::ext` API; the crates
here hold their contracts and a library that does not yet implement them.

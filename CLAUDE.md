# hql-extensions

Official HQL packages and the registry of known ones. The README says what
each package does and how one is registered; the design is HQL-0057 in the
hql repository, `doc/proposals/HQL-0057/`.

## What this repository is

- A Cargo workspace. Each crate under `crates/` is one package, named
  `hql-ext-<name>`, providing the module namespace `<name>` and nothing
  outside it. Its contracts live in `modules/` of the crate, written in
  HQL; its Rust implements them and declares nothing.
- `hql-packages.yaml` at the root is the registry. Every crate under
  `crates/` has an entry, and CI fails when one is missing or the file
  names a crate that is not there. A `known` entry points at a crate kept
  elsewhere and is added by pull request.

## Conventions

- A package is versioned and tagged on its own, `hql-ext-ml-v0.1.0`; the
  repository as a whole has no version. A tag is what the registry entry
  pins.
- A crate's `examples/` are checked HQL programs with the header hql's
  `examples/` use (`// status`, `// feature`, `// expected`); CI runs them
  against the `hql` versions the entry claims.
- Work goes on a short-lived branch and merges to `main`. Keep changes
  small; a package follows hql's language, not the other way round.
- No attribution of tools or assistants anywhere: not in commits, pull
  requests, code or docs.

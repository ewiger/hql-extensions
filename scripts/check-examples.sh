#!/bin/sh
# Run every example of every crate with the `hql` on PATH and hold it to its
# header: a `valid-now` example prints its `// expected:` value and checks to
# its `// expected-type:`; an `invalid` one fails with the `// kind:` it
# names. Each crate's `examples/.hql/config.toml` lists the crate under `[packages]
# paths`, so `hql` builds the package from this checkout on first use.
set -eu

header() {
    sed -n "s|^// $1: ||p" "$2" | head -n 1
}

failed=0
checked=0
errors=$(mktemp)
trap 'rm -f "$errors"' EXIT
for dir in crates/*/examples; do
    for file in "$dir"/*.hql; do
        [ -e "$file" ] || continue
        name=$(basename "$file")
        status=$(header status "$file")
        case "$status" in
        valid-now)
            expected=$(header expected "$file")
            expected_type=$(header expected-type "$file")
            if ! actual=$(cd "$dir" && hql run "$name" 2>"$errors"); then
                echo "FAIL $file: did not run"
                cat "$errors"
                failed=1
            elif [ "$actual" != "$expected" ]; then
                echo "FAIL $file: expected $expected, got $actual"
                failed=1
            elif [ -n "$expected_type" ] &&
                [ "$(cd "$dir" && hql check "$name" 2>/dev/null)" != "$expected_type" ]; then
                echo "FAIL $file: expected the type $expected_type"
                failed=1
            else
                echo "ok   $file"
            fi
            ;;
        invalid)
            kind=$(header kind "$file")
            if (cd "$dir" && hql run "$name") >/dev/null 2>"$errors"; then
                echo "FAIL $file: ran, and is declared invalid"
                failed=1
            elif ! grep -q "\[$kind\]" "$errors"; then
                echo "FAIL $file: expected a failure of kind $kind"
                cat "$errors"
                failed=1
            else
                echo "ok   $file"
            fi
            ;;
        *)
            echo "skip $file: status ${status:-none}"
            ;;
        esac
        checked=$((checked + 1))
    done
done
echo "$checked examples"
exit "$failed"

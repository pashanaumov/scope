// Minimal loader: maps .js specifiers to .ts counterparts so that
// `node --experimental-strip-types --loader ./loader.mjs --test src/foo.test.ts` works.

export async function resolve(specifier, context, nextResolve) {
  if (specifier.endsWith('.js')) {
    const tsSpecifier = `${specifier.slice(0, -3)}.ts`;
    try {
      const result = await nextResolve(tsSpecifier, context);
      return result;
    } catch {
      // fall through to original specifier
    }
  }
  return nextResolve(specifier, context);
}

import { existsSync } from 'node:fs';
import { basename, extname, join } from 'node:path';
import { Language, type Node, Parser } from 'web-tree-sitter';
import type { ScopeConfig } from './config-resolver.js';

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

export interface Chunk {
  id: string;
  filePath: string;
  language: string;
  startLine: number;
  endLine: number;
  header: string;
  context: string;
  body: string;
  embedText: string;
}

// ---------------------------------------------------------------------------
// Internal constants
// ---------------------------------------------------------------------------

const EXTENSION_TO_LANGUAGE: Record<string, string> = {
  '.ts': 'typescript',
  '.tsx': 'typescript',
  '.mts': 'typescript',
  '.cts': 'typescript',
  '.js': 'javascript',
  '.jsx': 'javascript',
  '.mjs': 'javascript',
  '.cjs': 'javascript',
  '.py': 'python',
  '.rs': 'rust',
  '.go': 'go',
  '.java': 'java',
  '.c': 'c',
  '.h': 'c',
  '.cpp': 'cpp',
  '.cc': 'cpp',
  '.cxx': 'cpp',
  '.hpp': 'cpp',
  '.rb': 'ruby',
};

// Node types to extract per language
const FUNCTION_NODE_TYPES: Record<string, string[]> = {
  typescript: [
    'function_declaration',
    'method_definition',
    'class_declaration',
    'lexical_declaration',
  ],
  javascript: [
    'function_declaration',
    'method_definition',
    'class_declaration',
    'lexical_declaration',
  ],
  python: ['function_definition', 'class_definition'],
  rust: ['function_item', 'impl_item'],
  go: ['function_declaration', 'method_declaration'],
  java: ['method_declaration', 'class_declaration'],
  c: ['function_definition'],
  cpp: ['function_definition'],
  ruby: ['method', 'class'],
};

// Import node types per language for context extraction
const IMPORT_NODE_TYPES: Record<string, string[]> = {
  typescript: ['import_statement'],
  javascript: ['import_statement', 'call_expression'], // call_expression for require()
  python: ['import_statement', 'import_from_statement'],
  rust: ['use_declaration'],
  go: ['import_declaration'],
  java: ['import_declaration'],
  c: ['preproc_include'],
  cpp: ['preproc_include'],
  ruby: ['call'], // require calls
};

// Parser cache: language → Parser instance
const parserCache = new Map<string, Parser>();
// Track whether Parser.init() has been called
let parserInitialized = false;
// Track warned-about missing grammars
const warnedGrammars = new Set<string>();

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function detectLanguage(filePath: string): string {
  const base = basename(filePath);
  // Check compound extensions first
  if (base.endsWith('.d.ts')) return 'typescript'; // will be filtered by ignored check
  if (base.endsWith('.min.js')) return 'javascript';
  const ext = extname(base);
  return EXTENSION_TO_LANGUAGE[ext] ?? 'unknown';
}

function isIgnored(filePath: string): boolean {
  const base = basename(filePath);
  if (base.endsWith('.d.ts') || base.endsWith('.min.js') || base.endsWith('.map')) return true;
  // Also handle .map as pure extension
  if (extname(base) === '.map') return true;
  return false;
}

function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

function makeChunkId(filePath: string, startLine: number, endLine: number): string {
  return `${filePath}:${startLine}:${endLine}`;
}

function buildEmbedText(header: string, context: string, body: string): string {
  const parts = [header];
  if (context) parts.push(context);
  parts.push(body);
  return parts.join('\n\n');
}

// ---------------------------------------------------------------------------
// Chunker class
// ---------------------------------------------------------------------------

type ChunkerConfig = Pick<
  ScopeConfig,
  'chunkMaxLines' | 'chunkOverlap' | 'chunkStrategy' | 'chunkMaxTokens'
> & {
  grammarsDir: string;
};

export class Chunker {
  private readonly config: ChunkerConfig;

  constructor(config: ChunkerConfig) {
    this.config = config;
  }

  async chunkFile(filePath: string, content: string): Promise<Chunk[]> {
    if (isIgnored(filePath)) return [];

    const language = detectLanguage(filePath);
    const lines = content.split('\n');

    if (this.config.chunkStrategy === 'function' && language !== 'unknown') {
      const chunks = await this.chunkByFunction(filePath, content, language, lines);
      if (chunks.length > 0) {
        return chunks.map((c) => this.applyTokenLimit(c));
      }
      // Fall through to sliding-window if no symbols found
    }

    return this.chunkBySlidingWindow(filePath, language, lines).map((c) => this.applyTokenLimit(c));
  }

  // -------------------------------------------------------------------------
  // Function strategy
  // -------------------------------------------------------------------------

  private async chunkByFunction(
    filePath: string,
    content: string,
    language: string,
    lines: string[],
  ): Promise<Chunk[]> {
    const parser = await this.getParser(language);
    if (!parser) {
      // Grammar not available → fall back
      return [];
    }

    const tree = parser.parse(content);
    if (!tree) return [];
    const nodeTypes = FUNCTION_NODE_TYPES[language] ?? [];
    const importTypes = new Set(IMPORT_NODE_TYPES[language] ?? []);

    // Collect top-of-file context (imports, max 20 lines of first contiguous import block)
    const context = extractImportContext(tree.rootNode, importTypes, lines, 20);

    // Walk top-level children and collect matching nodes
    const symbolNodes = collectSymbolNodes(tree.rootNode, nodeTypes);
    if (symbolNodes.length === 0) return [];

    const chunks: Chunk[] = [];
    for (const node of symbolNodes) {
      const startLine = node.startPosition.row + 1; // 1-indexed
      const endLine = node.endPosition.row + 1;
      const symbolName = extractSymbolName(node, language);
      const symbolKind = friendlyKind(node.type);
      const header = `${filePath} > ${symbolName} (${symbolKind})`;
      const body = lines.slice(node.startPosition.row, node.endPosition.row + 1).join('\n');

      chunks.push({
        id: makeChunkId(filePath, startLine, endLine),
        filePath,
        language,
        startLine,
        endLine,
        header,
        context,
        body,
        embedText: buildEmbedText(header, context, body),
      });
    }

    return chunks;
  }

  // -------------------------------------------------------------------------
  // Sliding-window strategy
  // -------------------------------------------------------------------------

  private chunkBySlidingWindow(filePath: string, language: string, lines: string[]): Chunk[] {
    const { chunkMaxLines, chunkOverlap } = this.config;
    const overlapLines = Math.floor(chunkMaxLines * chunkOverlap);
    const step = Math.max(1, chunkMaxLines - overlapLines);

    const chunks: Chunk[] = [];
    let start = 0;

    while (start < lines.length) {
      const end = Math.min(start + chunkMaxLines, lines.length);
      const startLine = start + 1;
      const endLine = end;
      const body = lines.slice(start, end).join('\n');
      const header = filePath;

      chunks.push({
        id: makeChunkId(filePath, startLine, endLine),
        filePath,
        language,
        startLine,
        endLine,
        header,
        context: '',
        body,
        embedText: buildEmbedText(header, '', body),
      });

      if (end === lines.length) break;
      start += step;
    }

    return chunks;
  }

  // -------------------------------------------------------------------------
  // Token limit enforcement
  // -------------------------------------------------------------------------

  private applyTokenLimit(chunk: Chunk): Chunk {
    const maxChars = this.config.chunkMaxTokens * 4;
    if (chunk.embedText.length <= maxChars) return chunk;

    process.stderr.write(
      `[chunker] WARNING: chunk ${chunk.id} exceeds token limit (${estimateTokens(chunk.embedText)} > ${this.config.chunkMaxTokens}), truncating\n`,
    );

    const truncatedEmbed = chunk.embedText.slice(0, maxChars);
    // Recompute body proportionally — truncate embedText and derive body from it
    const headerPart = chunk.header + (chunk.context ? `\n\n${chunk.context}\n\n` : '\n\n');
    const bodyStart = headerPart.length;
    const truncatedBody = truncatedEmbed.length > bodyStart ? truncatedEmbed.slice(bodyStart) : '';

    return {
      ...chunk,
      body: truncatedBody,
      embedText: truncatedEmbed,
    };
  }

  // -------------------------------------------------------------------------
  // Parser / grammar loading
  // -------------------------------------------------------------------------

  private async getParser(language: string): Promise<Parser | null> {
    if (parserCache.has(language)) return parserCache.get(language) as Parser;

    const wasmPath = join(this.config.grammarsDir, `tree-sitter-${language}.wasm`);
    if (!existsSync(wasmPath)) {
      if (!warnedGrammars.has(language)) {
        process.stderr.write(
          `[chunker] WARNING: grammar not found for '${language}' at ${wasmPath}, falling back to sliding-window\n`,
        );
        warnedGrammars.add(language);
      }
      return null;
    }

    if (!parserInitialized) {
      await Parser.init();
      parserInitialized = true;
    }

    try {
      const lang = await Language.load(wasmPath);
      const parser = new Parser();
      parser.setLanguage(lang);
      parserCache.set(language, parser);
      return parser;
    } catch (err) {
      process.stderr.write(`[chunker] WARNING: failed to load grammar for '${language}': ${err}\n`);
      return null;
    }
  }
}

// ---------------------------------------------------------------------------
// Tree-sitter helpers
// ---------------------------------------------------------------------------

function collectSymbolNodes(root: Node, nodeTypes: string[]): Node[] {
  const types = new Set(nodeTypes);
  const results: Node[] = [];

  function walk(node: Node): void {
    if (types.has(node.type)) {
      // For lexical_declaration (JS/TS arrow functions assigned to const/let)
      // only include if it contains an arrow_function
      if (node.type === 'lexical_declaration') {
        const hasArrow = hasDescendantOfType(node, 'arrow_function');
        if (hasArrow) {
          results.push(node);
          return; // don't recurse into it
        }
      } else {
        results.push(node);
        return; // don't recurse into top-level symbols
      }
    }
    for (const child of node.children) {
      walk(child);
    }
  }

  walk(root);
  return results;
}

function hasDescendantOfType(node: Node, type: string): boolean {
  if (node.type === type) return true;
  for (const child of node.children) {
    if (hasDescendantOfType(child, type)) return true;
  }
  return false;
}

function extractSymbolName(node: Node, _language: string): string {
  // Try common name-bearing child nodes
  const nameChild = node.childForFieldName('name');
  if (nameChild) return nameChild.text;

  // For lexical_declaration: look for variable_declarator name
  if (node.type === 'lexical_declaration') {
    for (const child of node.children) {
      if (child.type === 'variable_declarator') {
        const nameField = child.childForFieldName('name');
        if (nameField) return nameField.text;
      }
    }
  }

  // For method in Ruby
  const identChild = node.children.find(
    (c: Node) => c.type === 'identifier' || c.type === 'method_name',
  );
  if (identChild) return identChild.text;

  return '<anonymous>';
}

function friendlyKind(nodeType: string): string {
  const MAP: Record<string, string> = {
    function_declaration: 'function',
    function_definition: 'function',
    function_item: 'function',
    function: 'function',
    method_definition: 'method',
    method_declaration: 'method',
    method: 'method',
    method_declaration_item: 'method',
    function_declaration_item: 'function',
    class_declaration: 'class',
    class_definition: 'class',
    class: 'class',
    impl_item: 'impl',
    arrow_function: 'arrow',
    lexical_declaration: 'arrow',
    import_statement: 'import',
    function_declaration_2: 'function',
    method_declaration_2: 'method',
  };
  return MAP[nodeType] ?? nodeType;
}

function extractImportContext(
  root: Node,
  importTypes: Set<string>,
  lines: string[],
  maxLines: number,
): string {
  const importLines: number[] = [];

  // Collect row indices of import nodes at the top of the file
  for (const child of root.children) {
    if (importTypes.has(child.type)) {
      const row = child.startPosition.row;
      if (row < maxLines * 3) {
        // only scan near the top
        importLines.push(row);
      }
    } else if (importLines.length > 0) {
      // Stop at first non-import top-level node after seeing imports
      break;
    }
  }

  if (importLines.length === 0) return '';

  const lastImportRow = importLines[importLines.length - 1];
  const contextLines = lines.slice(0, Math.min(lastImportRow + 1, maxLines));
  return contextLines.join('\n');
}

// Run with Node: node tests/format_line_spacing.cjs
// TeX line breaks with any spacing (\[-6pt], \[0.5em]) must reach MathJax intact.
const fs = require('fs'), vm = require('vm');
const t = fs.readFileSync(require('path').join(__dirname, '..', 'templates', 'index.html'), 'utf8');
const start = t.indexOf('    function formatMessageSimplified(content)');
let depth = 0, i = t.indexOf('{', start), end;
for (; i < t.length; i++) { if (t[i] === '{') depth++; else if (t[i] === '}' && --depth === 0) { end = i + 1; break; } }
const ctx = { convertMarkdownTables: s => s }; vm.createContext(ctx); vm.runInContext(t.slice(start, end), ctx);
const BS = '\\';
const input = '**Step 1**\n\n' + BS + 'begin{align*}\nr &= ' + BS + 'frac{a}{b} ' + BS + BS + '[-6pt]\n&= 0.5 ' + BS + BS +
    '[6pt]\n&= 50' + BS + '%\n' + BS + 'end{align*}\n\nThen ' + BS + '[ x = 1 ' + BS + ']';
const out = ctx.formatMessageSimplified(input);
require('assert').ok(out.includes(BS + BS + '[-6pt]'), 'negative spacing kept as a line break');
require('assert').ok(out.includes(BS + BS + '[6pt]'), 'positive spacing kept');
require('assert').ok(out.includes(BS + '[ x = 1 ' + BS + ']'), 'display maths intact');
console.log('line spacing format checks passed');

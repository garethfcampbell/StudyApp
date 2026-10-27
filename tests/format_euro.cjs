// Run with Node: node tests/format_euro.cjs
// Euro signs: \unicode{x20AC} in maths mode (a raw euro crashes MathJax 3), a raw euro inside
// \text{} (where MathJax would print \unicode{x20AC} literally). Checked against MathJax 3 tex-svg.
const fs = require('fs'), vm = require('vm');
const t = fs.readFileSync(require('path').join(__dirname, '..', 'templates', 'index.html'), 'utf8');
const start = t.indexOf('    function formatMessageSimplified(content)');
let depth = 0, i = t.indexOf('{', start), end;
for (; i < t.length; i++) { if (t[i] === '{') depth++; else if (t[i] === '}' && --depth === 0) { end = i + 1; break; } }
const ctx = { convertMarkdownTables: s => s }; vm.createContext(ctx); vm.runInContext(t.slice(start, end), ctx);
const B = String.fromCharCode(92);
const samples = [
  B + '(' + B + 'text{€500 per year}' + B + ')',
  B + '(P = €500 + €20' + B + ')',
  B + '(' + B + 'text{Cost} = ' + B + 'text{' + B + 'unicode{x20AC}1,000}' + B + ')',
  B + '[' + B + 'text{NPV} = -€24{,}000 + ' + B + 'frac{€6{,}000}{1.1}' + B + ']',
  B + '(' + B + 'text{€5' + B + '% fee}' + B + ')'];
const out = samples.map(s => { const h = ctx.formatMessageSimplified(s); const m = h.match(/\[\(\[][\s\S]*?\[\)\]]/); return m ? m[0].slice(2, -2) : h; });
const assert = require('assert');
assert.ok(out[0].includes(B + 'text{€500 per year}'), out[0]);
assert.ok(out[1].includes('P = ' + B + 'unicode{x20AC}500 + ' + B + 'unicode{x20AC}20'), out[1]);
assert.ok(out[2].includes(B + 'text{Cost} = ' + B + 'text{€1,000}'), out[2]);
assert.ok(!out[3].includes('€') && out[3].includes(B + 'frac{' + B + 'unicode{x20AC}6'));
assert.ok(out[4].includes(B + 'text{€5}' + B + '%' + B + 'text{ fee}'), out[4]);
console.log('euro format checks passed');

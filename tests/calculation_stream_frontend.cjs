// Run with Node: node tests/calculation_stream_frontend.cjs
async function testCalculationFrontend(vm, assert, template) {
    const parser = template.slice(template.indexOf('async function streamSSE('),
        template.indexOf('// NEW SIMPLE CHAT SYSTEM'));
    const calculation = template.slice(template.indexOf('    let calculationStreamActive = false;'),
        template.indexOf('    async function streamQuickAction('));
    function setup(streams) {
        const state = { calls: 0, messages: [], answerInputs: 0, active: false };
        const element = () => ({
            classList: { remove() {}, add() { state.active = true; } },
            style: {}, appendChild() {}, remove() {}, scrollIntoView() {},
            innerHTML: '', scrollTop: 0, scrollHeight: 0
        });
        const messages = element();
        const context = {
            TextDecoder, console: { log() {}, warn() {}, error() {} },
            setTimeout(fn) { Promise.resolve().then(fn); return 0; }, clearTimeout() {}, clearInterval() {},
            currentProgressTimer: null,
            document: { getElementById: id => id === 'messages' ? messages : null,
                createElement: element, querySelectorAll: () => [] },
            formatMessageSimplified: text => text,
            addMessage: (role, text) => state.messages.push(text),
            aiTutor: { showCalculationAnswerInput() { state.answerInputs++; } },
            awaitCalculationVerification() {},
            fetch: async () => {
                const frames = streams[state.calls++];
                if (!frames) throw new Error('Unexpected retry');
                let index = 0;
                return {
                    ok: true,
                    headers: { get: name => name === 'content-type' ? 'text/event-stream' : 'test-request' },
                    body: { getReader: () => ({ read: async () => {
                        const frame = frames[index++];
                        if (frame instanceof Error) throw frame;
                        return frame === undefined ? { done: true } :
                            { done: false, value: new TextEncoder().encode(frame) };
                    } }) }
                };
            }
        };
        vm.createContext(context);
        vm.runInContext(parser + calculation, context);
        return { state, context, button: element() };
    }
    const question = 'data: "A calculation question"\n\ndata: [DONE]\n\n';
    let t = setup([[': waiting\n\ndata: [DONE]\n\n'], [question]]);
    await t.context.streamCalculationQuestion(t.button);
    assert.equal(t.state.calls, 2);
    assert.equal(t.state.answerInputs, 1);
    assert.equal(t.state.messages.length, 0);

    t = setup([[], []]);
    await t.context.streamCalculationQuestion(t.button);
    assert.equal(t.state.calls, 2);
    assert.equal(t.state.answerInputs, 0);
    assert.equal(t.state.messages.length, 1);
    assert.equal(t.state.active, false);

    t = setup([['data: {"error":"Empty","retryable":true}\n\n'], [question]]);
    await t.context.streamCalculationQuestion(t.button);
    assert.equal(t.state.calls, 2);
    assert.equal(t.state.answerInputs, 1);

    t = setup([[new Error('Network closed')], [question]]);
    await t.context.streamCalculationQuestion(t.button);
    assert.equal(t.state.calls, 2);

    t = setup([['data: "Partial question"\n\n', new Error('Network closed')]]);
    await t.context.streamCalculationQuestion(t.button);
    assert.equal(t.state.calls, 1); // Never regenerate a partially displayed question.
    assert.equal(t.state.answerInputs, 0);

    t = setup([[question]]);
    await Promise.all([t.context.streamCalculationQuestion(t.button),
        t.context.streamCalculationQuestion(t.button)]);
    assert.equal(t.state.calls, 1);

    t = setup([['data: {"error":"Not retryable","retryable":false}\n\n']]);
    await t.context.streamCalculationQuestion(t.button);
    assert.equal(t.state.calls, 1);
    assert.equal(t.state.messages[0], 'Not retryable');
    return '7 calculation frontend recovery checks passed';
}

if (typeof require !== 'undefined' && require.main === module) {
    testCalculationFrontend(require('node:vm'), require('node:assert/strict'),
        require('node:fs').readFileSync('templates/index.html', 'utf8'))
        .then(console.log).catch(error => { console.error(error); process.exitCode = 1; });
}

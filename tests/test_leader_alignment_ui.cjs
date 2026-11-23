// Run with: node tests/test_leader_alignment_ui.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const elements = new Map();
function element(id) {
  if (!elements.has(id)) elements.set(id, {
    hidden: false, disabled: false, textContent: '',
    classList: {toggle() {}}, addEventListener() {}
  });
  return elements.get(id);
}
const context = {
  window: {location: {origin: 'http://localhost:8000'}, addEventListener() {}},
  document: {getElementById: element},
  URL, WebSocket: {OPEN: 1}, console, setTimeout
};
vm.createContext(context);
const source = name => fs.readFileSync(path.join(__dirname, '../src/telerobot/web_ui/js', name), 'utf8');
vm.runInContext(source('websocket-manager.js'), context);
vm.runInContext(source('desktop-dashboard.js') + '\nwindow.dashboard = new DesktopDashboard();', context);
const dashboard = context.window.dashboard;
const manager = context.window.webSocketManager;
const sent = [];
manager.connectionStatus = 'connected';
manager.socket = {readyState: 1, send: data => sent.push(JSON.parse(data))};
dashboard.connected = true;
dashboard.render();
assert.equal(dashboard.matchButton.hidden, true); // unavailable on hardware/other modes

dashboard.onServerMessage({type: 'runtime_status', leader_alignment_available: true, dataset_configured: false});
assert.equal(dashboard.matchButton.hidden, false);
assert.equal(dashboard.matchButton.disabled, false); // no dataset needed

(async () => {
  await dashboard.matchLeader();
  await dashboard.matchLeader(); // repeated click is disabled while waiting
  assert.deepEqual(sent, [{command: 'align_simulation_to_leader'}]);
  assert.equal(dashboard.matchPending, true);
  dashboard.onServerMessage({type: 'leader_alignment_result', ok: true, message: 'Matched'});
  assert.equal(dashboard.matchPending, false);
  assert.equal(dashboard.matchStatus.textContent, 'Matched');
  dashboard.onServerMessage({type: 'runtime_status', recording: true});
  assert.equal(dashboard.matchButton.disabled, true);
  dashboard.onServerMessage({type: 'runtime_status', recording: false});
  dashboard.countingDown = true;
  dashboard.render();
  assert.equal(dashboard.matchButton.disabled, true);
  dashboard.countingDown = false;
  dashboard.onServerMessage({type: 'leader_alignment_result', ok: false, message: 'outside workspace'});
  assert.match(dashboard.matchStatus.textContent, /outside workspace/);
  await dashboard.matchLeader();
  dashboard.onConnectionChange('disconnected');
  assert.equal(dashboard.matchPending, false);
  assert.equal(dashboard.matchButton.disabled, true);
  assert.match(dashboard.matchStatus.textContent, /not confirmed/);
  console.log('Alignment UI checks passed.');
})().catch(error => { console.error(error); process.exitCode = 1; });

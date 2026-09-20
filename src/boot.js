import './style.css';
const status = document.getElementById('load-status');
let value = 0;
window.__loading = (label, amount) => {
  value = Math.max(value, Math.min(amount, 100));
  status.textContent = label;
  document.getElementById('load-progress').value = value;
  document.getElementById('load-progress').textContent = `${Math.round(value)}%`;
  document.getElementById('load-progress').setAttribute('aria-valuetext', `${Math.round(value)}% — ${label}`);
  document.getElementById('load-percent').textContent = `${Math.round(value)}%`;
};
window.__loadError = (error) => {
  console.error(error);
  document.getElementById('loading').classList.remove('hide');
  document.getElementById('loading').classList.add('failed');
  const graphics=/WebGL|graphics|Floating-point/i.test(String(error?.message||error));
  status.textContent = graphics ? 'The coastline could not open. Try reloading with hardware acceleration enabled.' : 'Part of the coastline could not load. Please try again.';
  document.getElementById('load-retry').hidden = false;
  document.getElementById('load-caption').textContent = graphics ? 'A browser with WebGL 2 is required.' : 'Check your connection, then reload the coastline.';
};
document.getElementById('load-retry').onclick = () => location.reload();
window.addEventListener('unhandledrejection', e => { if (!window.__ready) window.__loadError(e.reason); });
// Paint the shell before constructors start baking procedural materials.
requestAnimationFrame(() => requestAnimationFrame(() => {
  window.__loading('Preparing the coastline', 4);
  import('./main.js').catch(window.__loadError);
}));

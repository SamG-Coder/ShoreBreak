// Vite supplies the deployment base; Node-based authoring tools use the root.
export function assetUrl(path) {
  return (import.meta.env?.BASE_URL ?? '/') + path.replace(/^\/+/, '');
}

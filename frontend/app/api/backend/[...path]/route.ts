import { NextRequest } from "next/server";
export const dynamic = "force-dynamic";
async function proxy(
  request: NextRequest,
  context: { params: Promise<{ path: string[] }> },
) {
  const { path } = await context.params;
  if (!(path[0] === "health" || (path[0] === "api" && path[1] === "v1")))
    return new Response("Not found", { status: 404 });
  if (path.some((p) => p === ".." || p.includes("/")))
    return new Response("Invalid path", { status: 400 });
  const origin = request.headers.get("origin");
  const trustedOrigins = new Set([
    "http://127.0.0.1:4321",
    "http://localhost:4321",
  ]);
  const trustedHosts = new Set(["127.0.0.1:4321", "localhost:4321"]);
  if (!trustedHosts.has(request.headers.get("host") || ""))
    return Response.json({ detail: "不可信的本地主机" }, { status: 403 });
  if (request.method === "POST" && origin && !trustedOrigins.has(origin))
    return Response.json({ detail: "不可信的请求来源" }, { status: 403 });
  const base = process.env.MUSICTRACKER_BACKEND_URL || "http://127.0.0.1:8100";
  const headers = new Headers();
  for (const key of ["content-type", "idempotency-key", "last-event-id"]) {
    const value = request.headers.get(key);
    if (value) headers.set(key, value);
  }
  if (process.env.MUSICTRACKER_API_TOKEN)
    headers.set(
      "authorization",
      "Bearer " + process.env.MUSICTRACKER_API_TOKEN,
    );
  try {
    const response = await fetch(
      base +
        "/" +
        path.map(encodeURIComponent).join("/") +
        request.nextUrl.search,
      {
        method: request.method,
        headers,
        body: request.method === "POST" ? await request.text() : undefined,
        signal: request.signal,
        cache: "no-store",
      },
    );
    const output = new Headers();
    for (const key of [
      "content-type",
      "content-disposition",
      "cache-control",
    ]) {
      const value = response.headers.get(key);
      if (value) output.set(key, value);
    }
    return new Response(response.body, {
      status: response.status,
      headers: output,
    });
  } catch {
    return Response.json(
      { detail: "后端暂不可连接，请启动 MusicTracker 服务" },
      { status: 503 },
    );
  }
}
export const GET = proxy;
export const POST = proxy;

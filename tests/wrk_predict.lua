-- wrk script for POST /predict with varied payloads (same distribution as tests/load_api.py).
-- Usage: wrk -t4 -c100 -d30s --latency -s tests/wrk_predict.lua http://127.0.0.1:8000

local TYPES = {"CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"}
local N_PAYLOADS = 1000
local threads = {}

local function lognormal(mu, sigma)
  local u1, u2 = math.random(), math.random()
  local z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
  return math.exp(mu + sigma * z)
end

local function payload(i)
  local sender_balance = 0.0
  if math.random() < 0.5 then sender_balance = lognormal(10, 2) end
  return string.format(
    '{"transaction_id":"wrk-%d","type":"%s","amount":%.2f,"sender_id":"C%d","receiver_id":"C%d",'
      .. '"sender_balance":%.2f,"receiver_balance":%.2f}',
    i, TYPES[math.random(#TYPES)], lognormal(10, 1.5), math.random(1, 999999999),
    math.random(1, 999999999), sender_balance, lognormal(11, 2))
end

function setup(thread)
  thread:set("id", #threads + 1)
  table.insert(threads, thread)
end

function init(args)
  math.randomseed(42 + id)
  bodies = {}
  for i = 1, N_PAYLOADS do bodies[i] = payload(id * N_PAYLOADS + i) end
  idx = 0
  non_200 = 0
  wrk.method = "POST"
  wrk.path = "/predict"
  wrk.headers["Content-Type"] = "application/json"
end

function request()
  idx = idx % N_PAYLOADS + 1
  return wrk.format(nil, nil, nil, bodies[idx])
end

function response(status, headers, body)
  if status ~= 200 then non_200 = non_200 + 1 end
end

function done(summary, latency, requests)
  local bad = 0
  for _, t in ipairs(threads) do bad = bad + (t:get("non_200") or 0) end
  local errors = summary.errors.connect + summary.errors.read + summary.errors.write + summary.errors.timeout
  io.write(string.format(
    "RESULT req/s=%.0f p50=%.1fms p95=%.1fms p99=%.1fms max=%.1fms requests=%d non_200=%d socket_errors=%d\n",
    summary.requests / (summary.duration / 1e6), latency:percentile(50) / 1000, latency:percentile(95) / 1000,
    latency:percentile(99) / 1000, latency.max / 1000, summary.requests, bad, errors))
end

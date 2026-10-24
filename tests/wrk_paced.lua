-- Paced variant of tests/wrk_predict.lua (same payloads, same RESULT line) for a lower,
-- steadier load instead of "as fast as possible". Run from the project root:
--   wrk -t4 -c40 -d30s --latency -s tests/wrk_paced.lua http://127.0.0.1:8080 -- 10
-- Each connection waits <pace_ms> (here 10 ms) after a response before its next request,
-- so the rate is about connections * 1000 / (pace_ms + latency_ms). Read the achieved
-- rate from the RESULT line. Under saturation latency grows and the rate drops: this is
-- still a closed-loop generator, not a fixed arrival rate.
dofile("tests/wrk_predict.lua")

local base_init = init

function init(args)
  base_init(args)
  pace_ms = tonumber(args[1]) or 0
end

function delay()
  return pace_ms
end

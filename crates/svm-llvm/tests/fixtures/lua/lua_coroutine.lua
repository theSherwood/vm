-- In-house coroutine differential for the on-ramp. Exercises the whole coroutine
-- library surface that does NOT need the debug library (that is a separate slice):
-- create/resume/yield, status transitions, running/isyieldable, wrap, error
-- propagation, yield across pcall/xpcall (the continuation machinery),
-- coroutine.close + <close> variables, and a producer/filter/consumer pipeline.
-- A failing assert raises, so a clean exit means every assert held — identical to
-- native Lua.

print "testing coroutines (library slice)"

-- ── basics: multi-value transfer in both directions ──────────────────────────
do
  local co = coroutine.create(function(a, b)
    assert(a == 1 and b == 2)
    local c = coroutine.yield(a + b, a * b)     -- yields 3, 2
    assert(c == 10)
    local d, e = coroutine.yield(c * 2)         -- yields 20
    assert(d == 5 and e == 6)
    return "done", 42
  end)
  assert(coroutine.status(co) == "suspended")
  local ok, s, p = coroutine.resume(co, 1, 2)
  assert(ok and s == 3 and p == 2)
  ok, s = coroutine.resume(co, 10)
  assert(ok and s == 20)
  ok, s, p = coroutine.resume(co, 5, 6)
  assert(ok and s == "done" and p == 42)
  assert(coroutine.status(co) == "dead")
  ok, s = coroutine.resume(co)
  assert(not ok and string.find(s, "dead"))     -- cannot resume dead coroutine
end

-- ── coroutine.wrap ───────────────────────────────────────────────────────────
do
  local gen = coroutine.wrap(function()
    for i = 1, 3 do coroutine.yield(i * i) end
  end)
  assert(gen() == 1 and gen() == 4 and gen() == 9)

  -- wrap propagates an error as a plain (re-raised) error, not (false, msg)
  local w = coroutine.wrap(function() error("wrapfail") end)
  local ok, err = pcall(w)
  assert(not ok and string.find(err, "wrapfail"))
end

-- ── error out of create → (false, msg); coroutine goes dead ───────────────────
do
  local eco = coroutine.create(function() error({code = 7}) end)   -- non-string error value
  local ok, err = coroutine.resume(eco)
  assert(not ok and type(err) == "table" and err.code == 7)
  assert(coroutine.status(eco) == "dead")
end

-- ── running / isyieldable, main vs. inside a coroutine ───────────────────────
do
  local main, ismain = coroutine.running()
  assert(type(main) == "thread" and ismain == true)
  assert(coroutine.isyieldable() == false)
  assert(coroutine.isyieldable(main) == false)

  local co
  co = coroutine.create(function()
    local self, im = coroutine.running()
    assert(self == co and im == false)
    assert(coroutine.isyieldable() == true)
    coroutine.yield()
  end)
  assert(coroutine.resume(co))
end

-- ── status: "normal" when a coroutine has resumed another ────────────────────
do
  local outer
  outer = coroutine.create(function()
    local inner = coroutine.create(function()
      assert(coroutine.status(outer) == "normal")   -- outer is resuming us
      coroutine.yield()
    end)
    assert(coroutine.status(inner) == "suspended")
    assert(coroutine.resume(inner))
    assert(coroutine.status(inner) == "suspended")   -- inner yielded back
    return "ok"
  end)
  local ok, r = coroutine.resume(outer)
  assert(ok and r == "ok")
end

-- ── yield ACROSS a pcall boundary (yieldable pcall / continuations) ──────────
do
  local co = coroutine.wrap(function()
    local ok, r = pcall(function()
      local x = coroutine.yield("y1")
      assert(x == "resumed")
      return "pcall-ret"
    end)
    assert(ok and r == "pcall-ret")
    -- and across xpcall, with a message handler
    local ok2, r2 = xpcall(function()
      coroutine.yield("y2")
      error("boom")
    end, function(m) return "handled:" .. m end)
    assert(not ok2 and string.find(r2, "handled:") and string.find(r2, "boom"))
    coroutine.yield("y3")
  end)
  assert(co() == "y1")
  assert(co("resumed") == "y2")
  assert(co() == "y3")
end

-- ── coroutine.close: suspended → dead, and <close> vars run on close ──────────
do
  local co = coroutine.create(function() coroutine.yield(1); coroutine.yield(2) end)
  assert(coroutine.resume(co))
  assert(coroutine.status(co) == "suspended")
  local ok = coroutine.close(co)
  assert(ok == true and coroutine.status(co) == "dead")
  assert(coroutine.close(co) == true)              -- closing a dead coroutine is ok

  -- a to-be-closed variable inside a coroutine must run its __close when closed
  local closed = false
  local guard = setmetatable({}, {__close = function() closed = true end})
  local co2 = coroutine.create(function()
    local x <close> = guard
    coroutine.yield()
  end)
  assert(coroutine.resume(co2))
  assert(closed == false)                          -- still suspended, not yet closed
  assert(coroutine.close(co2) == true)
  assert(closed == true)                           -- __close fired on close
end

-- ── a classic producer / filter / consumer pipeline ──────────────────────────
do
  local function producer()
    return coroutine.wrap(function()
      for i = 1, 5 do coroutine.yield(i) end
    end)
  end
  local function doubler(prod)
    return coroutine.wrap(function()
      for v in prod do coroutine.yield(v * 2) end
    end)
  end
  local sum = 0
  for v in doubler(producer()) do sum = sum + v end
  assert(sum == (1+2+3+4+5) * 2)
end

-- ── two interleaved coroutines keep independent Lua stacks ───────────────────
do
  local function counter(n)
    return coroutine.wrap(function() for i = 1, n do coroutine.yield(i) end end)
  end
  local a, b = counter(3), counter(3)
  assert(a() == 1 and b() == 1 and a() == 2 and a() == 3 and b() == 2 and b() == 3)
end

print "ok"

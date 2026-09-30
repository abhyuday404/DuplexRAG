/* Guided-tour overlay injected into the live demo UI for the walkthrough video.
   Each step waits for a real UI condition (e.g. the first mid-sentence retrieval), then spotlights the
   element involved and shows a step card. Nothing is simulated: the app runs a normal real-time replay. */
(() => {
  const css = `
  #tour-spot { position: fixed; z-index: 9998; border-radius: 14px; pointer-events: none; opacity: 0;
    box-shadow: 0 0 0 4px rgba(124,92,255,.95), 0 0 0 9999px rgba(5,8,14,.55);
    transition: all .55s cubic-bezier(.2,.8,.2,1); }
  #tour-card { position: fixed; z-index: 9999; width: 430px; background: #ffffff; color: #14142b; border-radius: 16px;
    padding: 18px 20px 16px; box-shadow: 0 18px 50px rgba(0,0,0,.45); opacity: 0; transform: translateY(8px);
    transition: opacity .35s, transform .35s, left .55s cubic-bezier(.2,.8,.2,1), top .55s cubic-bezier(.2,.8,.2,1);
    font-family: "Noto Sans", "Liberation Sans", sans-serif; }
  #tour-card.show { opacity: 1; transform: none; }
  #tour-card .step { font: 700 12px "JetBrainsMono NF", monospace; letter-spacing: 2px; color: #6d28d9;
    text-transform: uppercase; }
  #tour-card h4 { font-size: 21px; margin: 6px 0 6px; line-height: 1.25; }
  #tour-card p { font-size: 15px; line-height: 1.45; color: #3c3c55; }
  #tour-card .bar { height: 4px; border-radius: 4px; background: #eee; margin-top: 12px; overflow: hidden; }
  #tour-card .bar i { display: block; height: 100%; background: linear-gradient(90deg,#7c5cff,#3fb6ff); width: 0; }
  #tour-banner { position: fixed; z-index: 9999; left: 50%; top: 14px; transform: translateX(-50%); background: #7c5cff;
    color: #fff; font: 700 16px "Noto Sans", sans-serif; padding: 8px 18px; border-radius: 999px;
    box-shadow: 0 8px 24px rgba(124,92,255,.4); }`;
  const style = document.createElement("style");
  style.textContent = css;
  document.head.appendChild(style);
  const spot = Object.assign(document.createElement("div"), { id: "tour-spot" });
  const card = Object.assign(document.createElement("div"), { id: "tour-card" });
  document.body.append(spot, card);
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  function place(el, pad = 8) {
    const r = el.getBoundingClientRect();
    Object.assign(spot.style, { left: r.left - pad + "px", top: r.top - pad + "px", width: r.width + 2 * pad + "px",
                                height: r.height + 2 * pad + "px", opacity: 1 });
    const W = innerWidth, H = innerHeight, cw = 430, ch = card.offsetHeight || 170;
    let x = r.right + 24, y = r.top;
    if (x + cw > W - 16) x = r.left - cw - 24;
    if (x < 16) { x = Math.min(Math.max(16, r.left), W - cw - 16); y = r.bottom + 20; }
    if (y + ch > H - 16) y = Math.max(16, r.top - ch - 20);
    if (y < 16) y = 16;
    Object.assign(card.style, { left: x + "px", top: y + "px" });
  }

  window.runTour = async (steps, banner) => {
    if (banner) {
      const b = Object.assign(document.createElement("div"), { id: "tour-banner", textContent: banner });
      document.body.appendChild(b);
    }
    let last = 0;
    for (let k = 0; k < steps.length; k++) {
      const s = steps[k];
      const t0 = Date.now();
      let el = null;
      while (Date.now() - t0 < (s.timeout || 90000)) {
        try { if (!s.wait || eval(s.wait)) { el = eval(s.target); if (el) break; } } catch (e) { /* not yet */ }
        await sleep(120);
      }
      const since = Date.now() - last;
      if (since < (steps[k - 1]?.min || 0)) await sleep((steps[k - 1]?.min || 0) - since);
      if (!el) continue;
      if (s.scroll) el.scrollIntoView({ block: "center" });
      card.classList.remove("show");
      await sleep(180);
      card.innerHTML = `<div class="step">Step ${k + 1} of ${steps.length}</div><h4>${s.title}</h4><p>${s.text}</p>
                        <div class="bar"><i></i></div>`;
      place(el, s.pad ?? 8);
      card.classList.add("show");
      const bar = card.querySelector(".bar i");
      bar.style.transition = `width ${(s.min || 3000) / 1000}s linear`;
      requestAnimationFrame(() => (bar.style.width = "100%"));
      last = Date.now();
      // keep the spotlight glued to a moving / growing element while this step is shown
      const follow = setInterval(() => { try { const e2 = eval(s.target); if (e2) place(e2, s.pad ?? 8); } catch (e) {} }, 400);
      s._follow = follow;
      if (k > 0 && steps[k - 1]._follow) clearInterval(steps[k - 1]._follow);
    }
    await sleep(steps.length ? (steps[steps.length - 1].min || 3000) : 0);
    window.__tourDone = true;
  };
})();

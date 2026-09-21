/* ClearPrep: minimal client script.
 * Only what the browser has to do itself: localStorage (history/profile), timers, and the opt-in Web Speech API.
 * All scoring happens on the server in Python. Nothing here sends data anywhere except to this app's own routes. */
(function () {
  "use strict";
  var KEY_HISTORY = "it.history.v1", KEY_PROFILE = "it.profile.v1";

  function read(key, fallback) {
    try { var v = localStorage.getItem(key); return v ? JSON.parse(v) : fallback; } catch (e) { return fallback; }
  }
  function write(key, value) {
    try { localStorage.setItem(key, JSON.stringify(value)); return true; } catch (e) { return false; }
  }
  window.itHistoryJSON = function () {
    // Send scores and labels only, never answer text, to render the dashboard.
    var h = read(KEY_HISTORY, []).map(function (r) {
      return { id: r.id, date: r.date, org: r.org, role: r.role, overall: r.overall, dims: r.dims, answered: r.answered, priorities: r.priorities, questions: (r.questions || []).map(function (q) { return { competency: q.competency, overall: q.overall }; }) };
    });
    return JSON.stringify(h);
  };
  window.itDate = function () { return (read(KEY_PROFILE, {}) || {}).interview_date || ""; };

  /* ---------- setup page ---------- */
  var FORM_FIELDS = ["org", "role", "family", "interview_date", "jd_text", "about_text", "interview_type", "seniority", "n", "difficulty", "time_limit"];
  var SAMPLE = {
    org: "Northwind Community Credit Union", role: "IT Security Analyst Intern", family: "auto", interview_type: "mixed", seniority: "internship", n: "6", difficulty: "mixed",
    jd_text: "IT Security Analyst Intern\n\nAbout the role\nThe intern supports our small security team in monitoring, documenting and improving how we protect member data.\n\nResponsibilities:\n- Monitor SIEM alerts in Splunk and help triage phishing reports\n- Document incidents and write clear summaries for non-technical staff\n- Assist with vulnerability scans and patch tracking\n\nRequirements:\n- Coursework in networking, operating systems or cybersecurity\n- Familiarity with Python or shell scripting\n- Strong communication skills and attention to detail\n- Ability to work with a team and manage deadlines",
    about_text: "Northwind Community Credit Union is a member-owned cooperative. We put member service first, protect member privacy, and invest in financial education for the community. We value integrity, transparency and teamwork."
  };
  function initSetup() {
    var form = document.getElementById("setup-form");
    if (!form) return;
    var saved = read(KEY_PROFILE, null);
    if (saved && !form.dataset.hasErrors) {
      FORM_FIELDS.forEach(function (f) {
        var el = form.elements[f];
        if (el && saved[f] != null && !el.value) el.value = saved[f];
      });
    }
    form.addEventListener("submit", function () {
      var out = {};
      FORM_FIELDS.forEach(function (f) { if (form.elements[f]) out[f] = form.elements[f].value; });
      write(KEY_PROFILE, out);
    });
    var sample = document.getElementById("fill-sample");
    if (sample) sample.addEventListener("click", function () {
      Object.keys(SAMPLE).forEach(function (k) { if (form.elements[k]) form.elements[k].value = SAMPLE[k]; });
    });
    // the JD/about counters
    document.querySelectorAll("textarea[data-max]").forEach(function (ta) {
      var out = document.getElementById(ta.id + "-count");
      var upd = function () { if (out) out.textContent = ta.value.length.toLocaleString() + " / " + Number(ta.dataset.max).toLocaleString(); };
      ta.addEventListener("input", upd); upd();
    });
  }

  /* ---------- countdown to the saved interview date (setup page banner and landing hero pill) ---------- */
  function initCountdown() {
    var box = document.getElementById("countdown");
    var saved = read(KEY_PROFILE, null);
    var d = saved && saved.interview_date;
    if (!box || !d) return;
    var t = new Date(), today = new Date(t.getFullYear(), t.getMonth(), t.getDate());
    var parts = d.split("-");
    var days = Math.round((new Date(+parts[0], +parts[1] - 1, +parts[2]) - today) / 86400000);  // whole calendar days, like the server
    if (isNaN(days) || days < 0) return;
    var txt = days === 0 ? "Your interview is today" : days + (days === 1 ? " day" : " days") + " until your interview" + (saved.org ? " at " + saved.org : "");
    box.querySelector("[data-days]").textContent = txt;
    box.hidden = false;
    box.classList.add("on");
  }

  /* ---------- landing page: reveal sections as they scroll into view ---------- */
  function initReveal() {
    var els = document.querySelectorAll("[data-reveal]");
    if (!els.length) return;
    var reduce = window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (!("IntersectionObserver" in window) || reduce) return;  // leave everything visible
    document.documentElement.classList.add("js-reveal");
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } });
    }, { threshold: 0.15, rootMargin: "0px 0px -6% 0px" });
    els.forEach(function (el) { io.observe(el); });
  }

  /* ---------- interview stage ---------- */
  var timerHandle = null;
  function stopTimer() { if (timerHandle) { clearInterval(timerHandle); timerHandle = null; } }

  function initStage(root) {
    var form = root.querySelector ? root.querySelector("#answer-form") : null;
    if (!form) { return; }
    if (form.dataset.itInit) return;
    form.dataset.itInit = "1";
    var ta = form.querySelector("textarea[name=answer]");
    var counter = form.querySelector("[data-counter]");
    var timerEl = document.getElementById("timer");
    var durationInput = form.querySelector("input[name=duration]");
    var spokenInput = form.querySelector("input[name=spoken]");
    var started = Date.now(), dictSeconds = 0, dictStart = null;

    function updCount() {
      if (!counter) return;
      var w = (ta.value.trim().match(/\S+/g) || []).length;
      counter.textContent = w + " words" + (w ? " · about " + Math.max(1, Math.round(w / 140 * 60)) + " s spoken" : "");
    }
    if (ta) { ta.addEventListener("input", updCount); updCount(); setTimeout(function () { ta.focus({ preventScroll: true }); }, 0); }

    stopTimer();
    if (timerEl) {
      var limit = parseInt(timerEl.dataset.limit || "0", 10);
      var tick = function () {
        var el = Math.floor((Date.now() - started) / 1000);
        var shown = limit ? Math.max(0, limit - el) : el;
        timerEl.textContent = Math.floor(shown / 60) + ":" + String(shown % 60).padStart(2, "0");
        if (limit) {
          timerEl.classList.toggle("warn", shown <= 15 && shown > 0);
          timerEl.classList.toggle("over", shown === 0);
          if (shown === 0) {
            stopTimer();
            if (ta && ta.value.trim()) { form.querySelector("[data-submit]").click(); }
            else timerEl.textContent = "0:00 · time’s up";
          }
        }
      };
      tick(); timerHandle = setInterval(tick, 500);
    }

    // send elapsed time (or active dictation time for spoken answers) with the answer
    form.addEventListener("htmx:configRequest", function (e) {
      if (dictStart) { dictSeconds += (Date.now() - dictStart) / 1000; dictStart = null; }
      var elapsed = (Date.now() - started) / 1000;
      if (durationInput) durationInput.value = (spokenInput && spokenInput.value === "1" && dictSeconds > 0 ? dictSeconds : elapsed).toFixed(1);
      e.detail.parameters.duration = durationInput ? durationInput.value : "";
      e.detail.parameters.spoken = spokenInput ? spokenInput.value : "";
    });

    // ---- opt-in voice: speechSynthesis to read the question, SpeechRecognition to dictate
    var speakBtn = root.querySelector("[data-speak]");
    if (speakBtn && "speechSynthesis" in window) {
      speakBtn.hidden = false;
      speakBtn.addEventListener("click", function () {
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(new SpeechSynthesisUtterance(speakBtn.dataset.speak));
      });
    }
    var dictBtn = root.querySelector("[data-dictate]");
    var SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    var note = root.querySelector("[data-voice-note]");
    if (dictBtn && SR && ta) {
      dictBtn.hidden = false; if (note) note.hidden = false;
      var rec = null, on = false;
      dictBtn.addEventListener("click", function () {
        if (on) { rec.stop(); return; }
        rec = new SR(); rec.lang = "en-US"; rec.continuous = true; rec.interimResults = false;
        var base = ta.value ? ta.value.replace(/\s+$/, "") + " " : "";
        rec.onresult = function (ev) {
          var t = ""; for (var i = 0; i < ev.results.length; i++) t += ev.results[i][0].transcript;
          ta.value = base + t.trim(); updCount();
        };
        rec.onstart = function () { on = true; dictStart = Date.now(); dictBtn.textContent = "■ Stop dictation"; if (spokenInput) spokenInput.value = "1"; };
        rec.onend = function () { on = false; if (dictStart) { dictSeconds += (Date.now() - dictStart) / 1000; dictStart = null; } dictBtn.textContent = "🎙 Dictate"; };
        rec.onerror = function () { on = false; dictBtn.textContent = "🎙 Dictate"; };
        rec.start();
      });
      ta.addEventListener("input", function () { if (!on && spokenInput && ta.value.trim().length < 3) spokenInput.value = ""; });
    }
  }

  /* ---------- report page: save the compact record to this browser ---------- */
  function initReport() {
    var node = document.getElementById("session-record");
    if (!node) return;
    var status = document.getElementById("save-status");
    try {
      var rec = JSON.parse(node.textContent);
      var hist = read(KEY_HISTORY, []).filter(function (r) { return r.id !== rec.id; });
      hist.push(rec);
      var ok = write(KEY_HISTORY, hist.slice(-60));
      if (status) status.textContent = ok ? "Saved to this browser (session " + hist.length + "). Only scores and labels are kept here, and the server stores nothing." : "Couldn’t save to this browser (storage is blocked). Use “Download Markdown” to keep this report.";
      var prof = read(KEY_PROFILE, {}) || {};
      var d = document.body.dataset.interviewDate;
      if (d && !prof.interview_date) { prof.interview_date = d; write(KEY_PROFILE, prof); }
    } catch (e) { if (status) status.textContent = "Couldn’t save this session to the browser."; }
  }

  function initProgress() {
    var clear = document.getElementById("clear-history");
    if (clear) clear.addEventListener("click", function () {
      if (confirm("Delete all saved practice history from this browser? This cannot be undone.")) { try { localStorage.removeItem(KEY_HISTORY); } catch (e) {} location.reload(); }
    });
    var exp = document.getElementById("export-history");
    if (exp) exp.addEventListener("click", function () {
      var blob = new Blob([JSON.stringify(read(KEY_HISTORY, []), null, 2)], { type: "application/json" });
      var a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "clearprep-history.json"; a.click();
    });
  }

  document.addEventListener("DOMContentLoaded", function () { initSetup(); initCountdown(); initReveal(); initReport(); initProgress(); });
  if (window.htmx) {
    // htmx.onLoad runs for the initial page and for every swapped-in fragment
    htmx.onLoad(function (el) { initStage(el); });
    // never fail silently: show a visible message if the server errors or can't be reached
    function showRequestError(msg) {
      var st = document.getElementById("stage"); if (!st) return;
      var old = document.getElementById("request-error"); if (old) old.remove();
      var n = document.createElement("div"); n.id = "request-error"; n.className = "notice err"; n.setAttribute("role", "alert"); n.textContent = msg;
      st.insertBefore(n, st.firstChild);
    }
    document.addEventListener("htmx:responseError", function (e) {
      var code = e.detail.xhr && e.detail.xhr.status;
      showRequestError(code === 400 || code === 413 ? "That session could not be read, so it can’t continue. Start a new interview from the home page. Your typed answer is still in the box if you want to copy it." : "Something went wrong on the server (error " + code + "). Your answer is still in the box. Try submitting again.");
    });
    document.addEventListener("htmx:sendError", function () { showRequestError("Couldn’t reach the server. Check your connection and try again. Your answer is still in the box."); });
    document.addEventListener("htmx:afterSwap", function () {
      var st = document.getElementById("stage");
      if (!st) return;
      st.scrollIntoView({ block: "start", behavior: "smooth" });
      var h = st.querySelector("[data-focus]"); if (h) h.focus({ preventScroll: true });
    });
  }
})();

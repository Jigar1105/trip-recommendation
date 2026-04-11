/**
 * itinerary_modal.js
 * Bharat Yatra — Enhanced Itinerary Modal with Source City
 *
 * Usage in index.html:
 *   1. <script src="/static/itinerary_modal.js"></script>  (before </body>)
 *   2. Plan Trip button: <button onclick="openItinModal({{ place | tojson }})">Plan Trip</button>
 */

(function () {
  'use strict';

  // ── AUTO-INJECT MODAL HTML INTO DOM ──────────────────────────
  function injectModal() {
    if (document.getElementById('itinModal')) return; // already exists

    const modalHTML = `
<div id="itinModal" style="
  display:none; position:fixed; inset:0; z-index:9999;
  background:rgba(0,0,0,0.65); backdrop-filter:blur(6px);
  align-items:center; justify-content:center; padding:16px;
">
  <div id="itinModalBox" style="
    background:#fff; border-radius:20px; width:100%; max-width:500px;
    max-height:92vh; overflow-y:auto;
    box-shadow:0 24px 80px rgba(0,0,0,0.3);
    animation: itinSlideUp 0.28s ease;
  ">
    <!-- Header -->
    <div style="
      background:linear-gradient(135deg,#1B4332,#2D5A27);
      padding:20px 22px 16px;
      border-radius:20px 20px 0 0;
      display:flex; align-items:flex-start; justify-content:space-between;
    ">
      <div>
        <div style="font-size:10px; color:rgba(255,255,255,.55); letter-spacing:2px; text-transform:uppercase; margin-bottom:5px;">
          AI TRIP PLANNER
        </div>
        <div id="itinModalTitle" style="font-size:17px; font-weight:700; color:#fff; line-height:1.3;">
          Plan Your Trip
        </div>
      </div>
      <button onclick="closeItinModal()" style="
        background:rgba(255,255,255,.12); border:none; color:#fff;
        width:34px; height:34px; border-radius:50%; cursor:pointer;
        font-size:16px; display:flex; align-items:center; justify-content:center;
        flex-shrink:0; margin-left:10px;
      ">✕</button>
    </div>

    <!-- Accent bar -->
    <div style="height:3px; background:linear-gradient(90deg,#FF6B35,#F4A261,transparent);"></div>

    <!-- Form -->
    <div style="padding:22px;">

      <!-- Source City -->
      <div style="margin-bottom:18px;">
        <label style="font-size:11px; font-weight:700; color:#1B4332;
          text-transform:uppercase; letter-spacing:1px; display:block; margin-bottom:8px;">
          📍 Aap Kidhar Se Travel Karenge?
        </label>
        <input
          type="text"
          id="itinSourceCity"
          placeholder="e.g. Mumbai, Delhi, Bangalore, Ahmedabad..."
          autocomplete="off"
          style="
            width:100%; padding:12px 15px;
            border:2px solid #E5E7EB; border-radius:11px;
            font-size:14px; outline:none;
            transition:border-color .2s, box-shadow .2s;
            font-family:inherit; color:#1A1A2E;
          "
        />
        <div id="itinSourceError" style="
          display:none; font-size:11px; color:#DC2626;
          margin-top:5px; font-weight:600;
        ">
          ⚠️ Starting city daalna zaroori hai
        </div>
        <div style="font-size:11px; color:#9CA3AF; margin-top:5px;">
          Is city se sabse sahi transport options dikhenge ✈️ 🚂 🚌 🚗
        </div>
      </div>

      <!-- Days + Style -->
      <div style="display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-bottom:16px;">
        <div>
          <label style="font-size:11px; font-weight:700; color:#1B4332;
            text-transform:uppercase; letter-spacing:1px; display:block; margin-bottom:8px;">
            📅 Kitne Din?
          </label>
          <select id="itinDays" style="
            width:100%; padding:11px 13px;
            border:2px solid #E5E7EB; border-radius:11px;
            font-size:14px; outline:none;
            background:#fff; cursor:pointer; font-family:inherit;
          ">
            <option value="2">2 Days</option>
            <option value="3" selected>3 Days</option>
            <option value="4">4 Days</option>
            <option value="5">5 Days</option>
            <option value="7">7 Days</option>
            <option value="10">10 Days</option>
          </select>
        </div>
        <div>
          <label style="font-size:11px; font-weight:700; color:#1B4332;
            text-transform:uppercase; letter-spacing:1px; display:block; margin-bottom:8px;">
            💎 Travel Style
          </label>
          <select id="itinStyle" style="
            width:100%; padding:11px 13px;
            border:2px solid #E5E7EB; border-radius:11px;
            font-size:14px; outline:none;
            background:#fff; cursor:pointer; font-family:inherit;
          ">
            <option value="budget">🪙 Budget</option>
            <option value="balanced" selected>⚖️ Balanced</option>
            <option value="luxury">💎 Luxury</option>
          </select>
        </div>
      </div>

      <!-- Style description -->
      <div id="itinStyleDesc" style="
        background:#F0FDF4; border:1px solid #BBF7D0; border-radius:10px;
        padding:10px 13px; font-size:12px; color:#166534;
        margin-bottom:16px; line-height:1.5;
      ">
        ⚖️ <strong>Balanced:</strong> Mid-range 3-star hotels, good restaurants, comfortable transport
      </div>

      <!-- Budget info -->
      <div style="
        background:#FFF8F0; border:1px solid #FDE8D0; border-radius:10px;
        padding:10px 13px; font-size:12px; color:#92400E; margin-bottom:20px;
        display:flex; align-items:center; justify-content:space-between;
      ">
        <span>💰 Max Budget per person</span>
        <span style="font-weight:700; font-size:14px;" id="itinBudgetDisplay">₹10,000</span>
      </div>

      <!-- Submit button -->
      <button
        id="itinSubmitBtn"
        onclick="submitItinerary()"
        style="
          width:100%; padding:14px;
          background:#FF6B35; color:#fff;
          border:none; border-radius:13px;
          font-size:15px; font-weight:700;
          cursor:pointer; transition:all .2s;
          font-family:inherit;
          display:flex; align-items:center; justify-content:center; gap:10px;
        "
      >
        <span id="itinBtnIcon">✨</span>
        <span id="itinBtnText">AI Itinerary Generate Karo</span>
      </button>

      <p style="font-size:11px; color:#9CA3AF; text-align:center; margin-top:10px; line-height:1.6;">
        AI generates transport prices, hotel options &amp; time-based schedule<br>
        with restaurants for every meal 🍽️
      </p>
    </div>
  </div>
</div>

<style>
@keyframes itinSlideUp {
  from { opacity:0; transform:translateY(28px) scale(.97); }
  to   { opacity:1; transform:translateY(0)   scale(1);   }
}
#itinSourceCity:focus {
  border-color:#FF6B35 !important;
  box-shadow:0 0 0 3px rgba(255,107,53,.12);
}
#itinSubmitBtn:hover {
  background:#E85D25 !important;
  transform:translateY(-1px);
  box-shadow:0 6px 20px rgba(255,107,53,.38);
}
#itinSubmitBtn:disabled {
  cursor:not-allowed; opacity:.78;
  transform:none !important; box-shadow:none !important;
}
</style>
`;
    document.body.insertAdjacentHTML('beforeend', modalHTML);
  }

  // ── STATE ────────────────────────────────────────────────────
  let _currentPlace = {};

  const styleDescs = {
    budget:   '🪙 <strong>Budget:</strong> Hostels/guesthouses, local dhabas, buses &amp; trains — maximum savings',
    balanced: '⚖️ <strong>Balanced:</strong> Mid-range 3-star hotels, good restaurants, comfortable transport',
    luxury:   '💎 <strong>Luxury:</strong> Premium 4-5 star resorts, fine dining, flights &amp; private transfers',
  };

  // ── HELPERS ──────────────────────────────────────────────────
  function el(id) { return document.getElementById(id); }

  function setLoading(loading) {
    const btn = el('itinSubmitBtn');
    btn.disabled = loading;
    if (loading) {
      el('itinBtnIcon').textContent = '⏳';
    } else {
      el('itinBtnIcon').textContent = '✨';
      el('itinBtnText').textContent = 'AI Itinerary Generate Karo';
    }
  }

  function cycleLoadingMsgs() {
    const msgs = [
      '✈️ Transport options dhoondh rahe hain...',
      '🏨 Best hotels select kar rahe hain...',
      '🗓 Day-by-day schedule bana rahe hain...',
      '🍽️ Famous restaurants add kar rahe hain...',
      '💰 Budget breakdown calculate ho rahi hai...',
      '✨ Almost done, thoda wait karo...',
    ];
    let idx = 0;
    el('itinBtnText').textContent = msgs[0];
    return setInterval(function () {
      idx = (idx + 1) % msgs.length;
      el('itinBtnText').textContent = msgs[idx];
    }, 2200);
  }

  // ── PUBLIC API ───────────────────────────────────────────────
  window.openItinModal = function (place) {
    injectModal();
    _currentPlace = place || {};

    // Set title
    const name = place['Place Name'] || place.place_name || 'This Place';
    el('itinModalTitle').textContent = 'Plan Trip to ' + name;

    // Set budget
    const budget = place['max_budget'] || place.max_budget || 10000;
    el('itinBudgetDisplay').textContent =
      '₹' + Number(budget).toLocaleString('en-IN');

    // Reset fields
    el('itinSourceCity').value = '';
    el('itinDays').value       = '3';
    el('itinStyle').value      = 'balanced';
    el('itinStyleDesc').innerHTML = styleDescs['balanced'];
    el('itinSourceError').style.display = 'none';
    setLoading(false);

    // Show
    const modal = el('itinModal');
    modal.style.display = 'flex';
    setTimeout(function () { el('itinSourceCity').focus(); }, 320);
  };

  window.closeItinModal = function () {
    const modal = el('itinModal');
    if (modal) modal.style.display = 'none';
  };

  window.submitItinerary = async function () {
    const sourceCity = el('itinSourceCity').value.trim();

    // Validate source city
    if (!sourceCity) {
      el('itinSourceCity').style.borderColor = '#EF4444';
      el('itinSourceCity').style.boxShadow   = '0 0 0 3px rgba(239,68,68,.15)';
      el('itinSourceError').style.display    = 'block';
      el('itinSourceCity').focus();
      setTimeout(function () {
        el('itinSourceCity').style.borderColor = '#E5E7EB';
        el('itinSourceCity').style.boxShadow   = '';
        el('itinSourceError').style.display    = 'none';
      }, 2500);
      return;
    }

    const place     = _currentPlace;
    const days      = parseInt(el('itinDays').value, 10);
    const style     = el('itinStyle').value;
    const budget    = place['max_budget'] || place.max_budget || 10000;
    const placeName = place['Place Name']  || place.place_name || '';
    const state     = place['State']       || place.state       || '';
    const placeType = place['Type']        || place.type        || '';
    const idealFor  = place['Ideal For']   || place.ideal_for   || 'all';

    setLoading(true);
    const interval = cycleLoadingMsgs();

    try {
      const resp = await fetch('/api/generate-itinerary', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          place_name:   placeName,
          state:        state,
          type:         placeType,
          days:         days,
          budget:       budget,
          travel_style: style,
          ideal_for:    idealFor,
          source_city:  sourceCity,
        }),
      });

      clearInterval(interval);
      const data = await resp.json();

      if (data.success && data.itinerary_id) {
        el('itinBtnIcon').textContent = '✅';
        el('itinBtnText').textContent = 'Itinerary ready! Redirect ho rahe hain...';
        setTimeout(function () {
          window.location.href = '/itinerary/' + data.itinerary_id;
        }, 800);
      } else {
        throw new Error(data.error || 'Unknown server error');
      }

    } catch (err) {
      clearInterval(interval);
      console.error('Itinerary generation error:', err);
      setLoading(false);
      el('itinBtnIcon').textContent = '⚠️';
      el('itinBtnText').textContent = 'Error: ' + err.message + ' — dobara try karo';
      setTimeout(function () {
        el('itinBtnIcon').textContent = '✨';
        el('itinBtnText').textContent = 'AI Itinerary Generate Karo';
      }, 4000);
    }
  };

  // ── EVENT LISTENERS ──────────────────────────────────────────
  // Style select change → update description
  document.addEventListener('change', function (e) {
    if (e.target && e.target.id === 'itinStyle') {
      const desc = el('itinStyleDesc');
      if (desc) desc.innerHTML = styleDescs[e.target.value] || '';
    }
  });

  // Close on backdrop click
  document.addEventListener('click', function (e) {
    const modal = el('itinModal');
    if (modal && e.target === modal) closeItinModal();
  });

  // Close on Escape key
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape') closeItinModal();
  });

  // Enter key in source city input → submit
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && e.target && e.target.id === 'itinSourceCity') {
      submitItinerary();
    }
  });

  // Auto-inject on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', injectModal);
  } else {
    injectModal();
  }

})();
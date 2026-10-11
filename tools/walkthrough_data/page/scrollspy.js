// Scrollspy: keep the sticky section nav's "active" tab in sync with whichever
// section is actually in view, rather than leaving it permanently on the first
// link (the nav didn't need this while it scrolled out of sight with the page;
// now that it's pinned via position: sticky, a static active link reads as broken).
(function () {
  const navEl = document.querySelector('.snav');
  const navLinks = Array.from(document.querySelectorAll('.snav a'));
  const sections = Array.from(document.querySelectorAll('main section[id]'));
  if (!navEl || !sections.length) return;

  function setActive(id) {
    navLinks.forEach(a => a.classList.toggle('active', a.getAttribute('href') === `#${id}`));
  }

  const observer = new IntersectionObserver((entries) => {
    const visible = entries.filter(e => e.isIntersecting);
    if (!visible.length) return;
    // If several sections are partly in view at once, the one nearest the top
    // (just under the sticky bar) is the one the reader is actually looking at.
    visible.sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
    setActive(visible[0].target.id);
  }, {
    // Offset for the sticky nav's own height, then only count a section as
    // "current" while its heading sits in the top third of the viewport.
    rootMargin: `-${navEl.offsetHeight + 1}px 0px -70% 0px`,
    threshold: 0,
  });

  sections.forEach((s) => observer.observe(s));
})();
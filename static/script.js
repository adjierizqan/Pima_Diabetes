document.addEventListener('DOMContentLoaded', () => {
    const sampleData = {
        Pregnancies: 3,
        Glucose: 128,
        BloodPressure: 72,
        SkinThickness: 28,
        Insulin: 105,
        BMI: 31.5,
        DiabetesPedigreeFunction: 0.52,
        Age: 45
    };

    const sampleBtn = document.getElementById('sample-btn');
    if (sampleBtn) {
        sampleBtn.addEventListener('click', () => {
            Object.entries(sampleData).forEach(([key, value]) => {
                const input = document.getElementById(key);
                if (input) {
                    input.value = value;
                    input.dispatchEvent(new Event('change'));
                    input.dispatchEvent(new Event('input'));
                }
            });
            animatePulse(sampleBtn);
        });
    }

    const reveals = document.querySelectorAll('.reveal');
    const observer = new IntersectionObserver(
        (entries) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('is-visible');
                    observer.unobserve(entry.target);
                }
            });
        },
        {
            threshold: 0.15,
            rootMargin: '0px 0px -50px 0px'
        }
    );

    reveals.forEach((element) => observer.observe(element));

    const progressBar = document.querySelector('.progress-bar[data-progress]');
    if (progressBar) {
        const target = parseFloat(progressBar.dataset.progress || '0');
        requestAnimationFrame(() => {
            progressBar.style.width = `${Math.min(Math.max(target, 0), 100)}%`;
        });
    }

    function animatePulse(element) {
        anime({
            targets: element,
            scale: [1, 1.05, 1],
            duration: 400,
            easing: 'easeInOutQuad'
        });
    }
});

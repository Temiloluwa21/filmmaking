document.addEventListener('DOMContentLoaded', () => {

    // --- State Management ---
    let currentUser = null; 

    // --- DOM Elements ---
    const navbar = document.getElementById('navbar');
    const navLinks = document.querySelectorAll('.nav-btn, .nav-links a[data-target]');
    const views = document.querySelectorAll('.view');
    const authOnlyLinks = document.querySelectorAll('.auth-only');
    const logoutBtn = document.getElementById('logout-btn');
    const toast = document.getElementById('toast');

    // Forms
    const loginForm = document.getElementById('login-form');
    const registerForm = document.getElementById('register-form');

    // Summarizer Dashboard Elements
    const videoInput = document.getElementById('video-input');
    const queryInput = document.getElementById('query-input');
    const summarizeBtn = document.getElementById('summarize-btn');
    const loadingSection = document.getElementById('loading-section');
    const resultSection = document.getElementById('result-section');
    const uploadSection = document.getElementById('upload-section');
    const loadingText = document.getElementById('loading-text');
    const progressFill = document.getElementById('progress-fill');
    const summaryPlayer = document.getElementById('summary-player');
    const downloadLink = document.getElementById('download-link');
    const fileDropArea = document.getElementById('file-drop-area');
    const fileMsg = document.querySelector('.file-msg');
    const newVideoBtn = document.getElementById('new-video-btn');

    // --- 1. SPA Routing Logic ---
    function navigateTo(targetViewId) {
        if ((targetViewId === 'dashboard-view') && !currentUser) {
            showToast("Please log in to access this page.", true);
            navigateTo('login-view');
            return;
        }

        views.forEach(view => {
            view.classList.remove('active-view');
            view.classList.add('hidden-view');
        });

        const targetView = document.getElementById(targetViewId);
        if (targetView) {
            targetView.classList.remove('hidden-view');
            targetView.classList.add('active-view');
        }

        document.querySelectorAll('.nav-links a').forEach(link => {
            link.classList.remove('active');
            if(link.getAttribute('data-target') === targetViewId) {
                link.classList.add('active');
            }
        });
    }

    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const target = link.getAttribute('data-target');
            if(target) navigateTo(target);
        });
    });

    // --- 2. Mock Authentication Logic ---
    function updateAuthState(userEmail) {
        currentUser = userEmail;
        if(currentUser) {
            navbar.classList.remove('hidden');
            authOnlyLinks.forEach(link => link.classList.remove('hidden'));
            navigateTo('dashboard-view');
            showToast(`Welcome back, ${currentUser}!`);
        } else {
            navbar.classList.add('hidden');
            authOnlyLinks.forEach(link => link.classList.add('hidden'));
            navigateTo('home-view');
        }
    }

    loginForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const email = document.getElementById('login-email').value;
        updateAuthState(email);
    });

    registerForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const email = document.getElementById('reg-email').value;
        const name = document.getElementById('reg-name').value;
        showToast(`Account created for ${name}!`);
        updateAuthState(email);
    });

    logoutBtn.addEventListener('click', (e) => {
        e.preventDefault();
        currentUser = null;
        updateAuthState(null);
        showToast("Logged out successfully.");
    });


    // --- 3. Dashboard UI Interactions ---
    fileDropArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        fileDropArea.style.borderColor = '#6366f1';
        fileDropArea.style.backgroundColor = 'rgba(99, 102, 241, 0.1)';
    });

    fileDropArea.addEventListener('dragleave', (e) => {
        e.preventDefault();
        fileDropArea.style.borderColor = 'rgba(255, 255, 255, 0.2)';
        fileDropArea.style.backgroundColor = 'transparent';
    });

    fileDropArea.addEventListener('drop', (e) => {
        e.preventDefault();
        fileDropArea.style.borderColor = 'rgba(255, 255, 255, 0.2)';
        fileDropArea.style.backgroundColor = 'transparent';
        if (e.dataTransfer.files.length > 0) {
            videoInput.files = e.dataTransfer.files;
            fileMsg.textContent = e.dataTransfer.files[0].name;
        }
    });

    videoInput.addEventListener('change', () => {
        if (videoInput.files.length > 0) {
            fileMsg.textContent = videoInput.files[0].name;
        }
    });

    newVideoBtn.addEventListener('click', () => {
        resultSection.classList.add('hidden');
        uploadSection.classList.remove('hidden');
        fileMsg.textContent = "Drag & Drop .mp4 Video Here";
        videoInput.value = '';
        progressFill.style.width = '0%';
    });


    // --- 4. Summarization Polling Logic ---

    async function pollStatus(taskId) {
        const interval = setInterval(async () => {
            try {
                const res = await fetch(`/api/status/${taskId}`);
                if (!res.ok) return;
                const data = await res.json();
                
                // Update Progress UI
                progressFill.style.width = `${data.progress}%`;
                loadingText.textContent = data.status;

                if (data.progress === 100 && data.file_id) {
                    clearInterval(interval);
                    
                    // Direct URL mapping to avoid 'WinError 10054' by letting browser handle streaming
                    const videoUrl = `/api/download/${data.file_id}`;
                    
                    loadingSection.classList.add('hidden');
                    resultSection.classList.remove('hidden');
                    summaryPlayer.src = videoUrl;
                    summaryPlayer.load();
                    downloadLink.href = videoUrl;
                    downloadLink.download = "ai_summary.mp4";
                    showToast("Summary Generated Successfully!");
                } else if (data.status.startsWith("Error")) {
                    clearInterval(interval);
                    throw new Error(data.status);
                }
            } catch (error) {
                console.error(error);
                clearInterval(interval);
                showToast(error.message || "Processing failed.", true);
                loadingSection.classList.add('hidden');
                uploadSection.classList.remove('hidden');
            }
        }, 1500);
    }

    summarizeBtn.addEventListener('click', async () => {
        if (videoInput.files.length === 0) {
            showToast("Please upload a video first.", true);
            return;
        }
        
        const formData = new FormData();
        formData.append("video", videoInput.files[0]);
        if (queryInput.value.trim()) formData.append("query", queryInput.value.trim());

        uploadSection.classList.add('hidden');
        loadingSection.classList.remove('hidden');
        progressFill.style.width = '2%';
        loadingText.textContent = "Uploading video to secure container...";

        try {
            const response = await fetch('/api/summarize', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error("Upload failed.");
            
            const data = await response.json();
            pollStatus(data.task_id); // Start polling for real-time progress
            
        } catch (error) {
            console.error(error);
            showToast("Upload failed. Ensure backend is running.", true);
            loadingSection.classList.add('hidden');
            uploadSection.classList.remove('hidden');
        }
    });

    function showToast(message, isError = false) {
        toast.textContent = message;
        toast.style.background = isError ? '#ef4444' : '#10b981';
        toast.classList.remove('hidden');
        setTimeout(() => toast.classList.add('hidden'), 3000);
    }
});

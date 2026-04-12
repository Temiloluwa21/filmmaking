document.addEventListener('DOMContentLoaded', () => {

    // --- State Management ---
    let currentUser = null; 
    let libraryItems = JSON.parse(localStorage.getItem('aiLibraryItems')) || [];

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
    const summaryPlayer = document.getElementById('summary-player');
    const downloadLink = document.getElementById('download-link');
    const fileDropArea = document.getElementById('file-drop-area');
    const fileMsg = document.querySelector('.file-msg');
    const newVideoBtn = document.getElementById('new-video-btn');
    const saveLibraryBtn = document.getElementById('save-library-btn');
    const libraryGallery = document.getElementById('library-gallery');

    // --- 1. SPA Routing Logic ---
    function navigateTo(targetViewId) {
        // Enforce Authentication
        if ((targetViewId === 'dashboard-view' || targetViewId === 'library-view') && !currentUser) {
            showToast("Please log in to access this page.", true);
            navigateTo('login-view');
            return;
        }

        // Hide all views
        views.forEach(view => {
            view.classList.remove('active-view');
            view.classList.add('hidden-view');
        });

        // Show target view
        const targetView = document.getElementById(targetViewId);
        if (targetView) {
            targetView.classList.remove('hidden-view');
            targetView.classList.add('active-view');
        }

        // Update Navbar Activity State
        document.querySelectorAll('.nav-links a').forEach(link => {
            link.classList.remove('active');
            if(link.getAttribute('data-target') === targetViewId) {
                link.classList.add('active');
            }
        });
    }

    // Attach click listeners to all navigational buttons/links
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
        // Mocking login immediately
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
    });


    // --- 4. The Extent of Summarization Logic & LocalStorage ---

    function renderLibrary() {
        const emptyState = document.querySelector('.empty-state');
        // Clear current elements, keeping the empty state div if exists
        libraryGallery.innerHTML = '';
        if (libraryItems.length === 0) {
            if (emptyState) libraryGallery.appendChild(emptyState);
            return;
        }

        libraryItems.forEach(item => {
            const card = document.createElement('div');
            card.className = 'library-card';
            card.style.background = 'rgba(30, 41, 59, 0.8)';
            card.style.padding = '1rem';
            card.style.borderRadius = '12px';
            card.style.border = '1px solid rgba(255,255,255,0.1)';
            
            card.innerHTML = `
                <div style="width:100%; height:150px; border-radius:8px; margin-bottom:0.5rem; background:linear-gradient(45deg, #4f46e5, #c084fc); display:flex; align-items:center; justify-content:center; color:white; font-weight:800;">🎥 ${item.videoName}</div>
                <p style="font-size:0.9rem; color:white;">Query: "${item.query}"</p>
                <p style="font-size:0.8rem; color:#94a3b8;">Saved: ${item.date}</p>
            `;
            libraryGallery.prepend(card);
        });
    }

    // Load library immediately
    renderLibrary();

    saveLibraryBtn.addEventListener('click', () => {
        if(!summaryPlayer.src) return;
        
        let vName = "Video Snippet";
        if (videoInput.files.length > 0) vName = videoInput.files[0].name;

        const newItem = {
            videoName: vName,
            query: queryInput.value,
            date: new Date().toLocaleDateString()
        };

        libraryItems.push(newItem);
        localStorage.setItem('aiLibraryItems', JSON.stringify(libraryItems));
        
        renderLibrary();
        showToast("Saved to My Library Permanently!");
    });

    summarizeBtn.addEventListener('click', async () => {
        if (videoInput.files.length === 0) {
            showToast("Please upload a video first.", true);
            return;
        }
        
        const videoFile = videoInput.files[0];
        const query = queryInput.value.trim();

        const formData = new FormData();
        formData.append("video", videoFile);
        if (query) {
            formData.append("query", query);
        }

        uploadSection.classList.add('hidden');
        loadingSection.classList.remove('hidden');

        try {
            const response = await fetch('/api/summarize', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error("Summarization failed.");
            }

            const videoBlob = await response.blob();
            const videoUrl = URL.createObjectURL(videoBlob);
            
            loadingSection.classList.add('hidden');
            resultSection.classList.remove('hidden');
            
            summaryPlayer.src = videoUrl;
            downloadLink.href = videoUrl;
            downloadLink.download = "ai_summary.mp4";
            
            showToast("Summary Generated Successfully!");
        } catch (error) {
            console.error(error);
            showToast("An error occurred during processing. Please ensure your backend is running.", true);
            loadingSection.classList.add('hidden');
            uploadSection.classList.remove('hidden');
        }
    });

    // --- Helper ---
    function showToast(message, isError = false) {
        toast.textContent = message;
        toast.style.background = isError ? '#ef4444' : '#10b981';
        toast.classList.remove('hidden');
        setTimeout(() => toast.classList.add('hidden'), 3000);
    }
});

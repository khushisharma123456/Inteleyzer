export class Sidebar extends HTMLElement {
    constructor() {
        super();
        this.collapsed = false;
        this.role = localStorage.getItem('user_role') || 'doctor';
    }

    connectedCallback() {
        this.render();
        this.addEventListeners();
    }

    addEventListeners() {
        const toggleBtn = this.querySelector('#toggle-sidebar');
        if (toggleBtn) {
            toggleBtn.addEventListener('click', () => {
                this.collapsed = !this.collapsed;
                document.body.classList.toggle('sidebar-collapsed', this.collapsed);
                this.render();
            });
        }
    }

    getNavItems() {
        // Flask Routes
        const doctorItems = [
            { icon: 'home', text: 'Overview', link: '/doctor/dashboard' },
            { icon: 'users', text: 'Patients', link: '/doctor/patients' },
            { icon: 'file-plus-2', text: 'Report Adverse Effect', link: '/doctor/report' },
            { icon: 'alert-triangle', text: 'Safety Alerts', link: '/doctor/alerts' },
            { icon: 'shield-alert', text: 'Drug Advisories', link: '/doctor/warnings' },
        ];

        const pharmaItems = [
            { icon: 'activity', text: 'Overview', link: '/pharma/dashboard' },
            { icon: 'pill', text: 'Drug Portfolio', link: '/pharma/drugs' },
            { icon: 'file-text', text: 'Reports', link: '/pharma/reports' },
            { icon: 'user-check', text: 'Patient Recall', link: '/pharma/patient-recall' },
        ];

        const pharmacyItems = [
            { icon: 'home', text: 'Dashboard', link: '/pharmacy/dashboard' },
            { icon: 'file-text', text: 'Reports', link: '/pharmacy/reports' },
            { icon: 'package', text: 'Dispensing Logs', link: '/pharmacy/dispensing-logs' },
            { icon: 'alert-triangle', text: 'Alerts', link: '/pharmacy/alerts' },
        ];

        if (this.role === 'pharma') {
            return pharmaItems;
        } else if (this.role === 'pharmacy') {
            return pharmacyItems;
        } else {
            return doctorItems;
        }
    }

    render() {
        const items = this.getNavItems();
        const currentPath = window.location.pathname;

        this.innerHTML = `
            <style>
                aside {
                    width: var(--sidebar-width);
                    height: 100vh;
                    background-color: var(--secondary);
                    border-right: 1px solid #E2E8F0;
                    position: fixed;
                    left: 0;
                    top: 0;
                    display: flex;
                    flex-direction: column;
                    transition: width 0.3s ease;
                    z-index: 100;
                }
                
                body.sidebar-collapsed aside {
                    width: var(--sidebar-collapsed-width);
                }

                .logo-area {
                    padding: 1.5rem;
                    display: flex;
                    align-items: center;
                    gap: 1rem;
                    border-bottom: 1px solid #F1F5F9;
                }

                .logo-icon {
                    width: 32px;
                    height: 32px;
                    background-color: var(--primary);
                    border-radius: 8px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: white;
                }

                .app-title {
                    font-weight: 700;
                    color: var(--primary-dark);
                    white-space: nowrap;
                    overflow: hidden;
                }
                
                body.sidebar-collapsed .app-title {
                    display: none;
                }

                nav {
                    flex: 1;
                    padding: 1rem 0;
                }

                .nav-item {
                    display: flex;
                    align-items: center;
                    padding: 0.75rem 1.5rem;
                    color: var(--text-muted);
                    text-decoration: none;
                    transition: all 0.2s;
                    gap: 1rem;
                }

                .nav-item:hover, .nav-item.active {
                    background-color: var(--secondary);
                    color: var(--primary-dark);
                    border-right: 3px solid var(--primary);
                }

                .nav-text {
                    white-space: nowrap;
                }
                
                body.sidebar-collapsed .nav-text {
                    display: none;
                }

                .user-profile {
                    padding: 1rem;
                    border-top: 1px solid #F1F5F9;
                    display: flex;
                    align-items: center;
                    gap: 1rem;
                }

                .footer-section {
                    margin-top: auto;
                    border-top: 1px solid #F1F5F9;
                    padding: 1rem 0;
                }

                .settings-btn {
                    display: flex;
                    align-items: center;
                    padding: 0.75rem 1.5rem;
                    color: var(--text-muted);
                    text-decoration: none;
                    transition: all 0.2s;
                    gap: 1rem;
                }

                .settings-btn:hover {
                    background-color: var(--secondary);
                    color: var(--primary-dark);
                    border-right: 3px solid var(--primary);
                }

                .logout-btn {
                    display: flex;
                    align-items: center;
                    padding: 0.75rem 1.5rem;
                    color: #DC2626;
                    text-decoration: none;
                    transition: all 0.2s;
                    gap: 1rem;
                    cursor: pointer;
                    border: none;
                    background: none;
                    width: 100%;
                    font-size: 1rem;
                    font-family: inherit;
                }

                .logout-btn:hover {
                    background-color: #FEF2F2;
                    color: #991B1B;
                    border-right: 3px solid #DC2626;
                }

                .doctor-name {
                    padding: 0.75rem 1.5rem;
                    color: var(--text-muted);
                    font-size: 0.9rem;
                    text-align: center;
                    white-space: nowrap;
                    overflow: hidden;
                    text-overflow: ellipsis;
                    background-color: white;
                    border-radius: 8px;
                    margin: 0.5rem 1rem;
                    font-weight: 500;
                }
                
                .doctor-email {
                    padding: 0.5rem 1.5rem;
                    color: var(--text-muted);
                    font-size: 0.8rem;
                    text-align: center;
                    white-space: nowrap;
                    overflow: hidden;
                    text-overflow: ellipsis;
                    margin: 0 1rem 0.5rem 1rem;
                }
                
                body.sidebar-collapsed .doctor-name {
                    font-size: 0.75rem;
                    padding: 0.5rem 0.5rem;
                    margin: 0.5rem 0.25rem;
                }
                
                body.sidebar-collapsed .doctor-email {
                    display: none;
                }
            </style>
            
            <aside>
                <div class="logo-area">
                    <img src="/static/images/logo.jpeg" alt="Inteleyzer Logo" style="width: 32px; height: 32px; border-radius: 8px; object-fit: cover;">
                    <span class="app-title">Inteleyzer</span>
                </div>

                <nav>
                    ${items.map(item => `
                        <a href="${item.link}" class="nav-item ${currentPath.includes(item.link) ? 'active' : ''}">
                            <i data-lucide="${item.icon}"></i>
                            <span class="nav-text">${item.text}</span>
                        </a>
                    `).join('')}
                </nav>

                <div class="footer-section">
                    <a href="${this.role === 'pharmacy' ? '/pharmacy/settings' : this.role === 'pharma' ? '/pharma/settings' : '/doctor/settings'}" class="settings-btn">
                        <i data-lucide="settings"></i>
                        <span class="nav-text">Settings</span>
                    </a>
                    <button onclick="import('/static/js/auth.js').then(m => m.Auth.logout())" class="logout-btn">
                        <i data-lucide="log-out"></i>
                        <span class="nav-text">Logout</span>
                    </button>
                    <div class="doctor-name">${localStorage.getItem('user_name') || 'User'}</div>
                    <div class="doctor-email">${localStorage.getItem('user_email') || ''}</div>
                </div>
            </aside>
        `;

        if (window.lucide) {
            window.lucide.createIcons();
        }
    }
}

customElements.define('app-sidebar', Sidebar);

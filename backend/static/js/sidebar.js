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
            { icon: 'home', text: 'Dashboard', link: '/doctor/dashboard' },
            { icon: 'users', text: 'My Patients', link: '/doctor/patients' },
            { icon: 'file-plus-2', text: 'Report Experience', link: '/doctor/report' },
            { icon: 'alert-triangle', text: 'Alerts', link: '/doctor/alerts' },
            { icon: 'shield-alert', text: 'Drug Warnings', link: '/doctor/warnings' },
            { icon: 'bar-chart-3', text: 'Analysis', link: '/doctor/analysis' },
        ];

        const pharmaItems = [
            { icon: 'activity', text: 'Overview', link: '/pharma/dashboard' },
            { icon: 'pill', text: 'Drug Portfolio', link: '/pharma/drugs' },
            { icon: 'file-text', text: 'Reports', link: '/pharma/reports' },
            { icon: 'bar-chart-3', text: 'Analysis', link: '/pharma/analysis' },
        ];

        const pharmacyItems = [
            { icon: 'home', text: 'Dashboard', link: '/pharmacy/dashboard' },
            { icon: 'clipboard-list', text: 'ADR Reports', link: '/pharmacy/reports' },
            { icon: 'file-plus', text: 'Report ADR', link: '/pharmacy/report' },
            { icon: 'bell', text: 'Safety Alerts', link: '/pharmacy/alerts' },
        ];

        if (this.role === 'pharma') return pharmaItems;
        if (this.role === 'pharmacy') return pharmacyItems;
        return doctorItems;
    }

    render() {
        const items = this.getNavItems();
        const currentPath = window.location.pathname;

        this.innerHTML = `
            <style>
                aside {
                    width: var(--sidebar-width);
                    height: 100vh;
                    background-color: var(--white);
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

                .logout-btn {
                    margin-top: auto;
                    color: var(--risk-high);
                    cursor: pointer;
                }
            </style>
            
            <aside>
                <div class="logo-area">
                    <div class="logo-icon">Rx</div>
                    <span class="app-title">MedSafe</span>
                </div>

                <nav>
                    ${items.map(item => `
                        <a href="${item.link}" class="nav-item ${currentPath.includes(item.link) ? 'active' : ''}">
                            <i data-lucide="${item.icon}"></i>
                            <span class="nav-text">${item.text}</span>
                        </a>
                    `).join('')}
                </nav>

                <div class="user-profile">
                     <a onclick="import('/static/js/auth.js').then(m => m.Auth.logout())" class="nav-item logout-btn">
                        <i data-lucide="log-out"></i>
                        <span class="nav-text">Logout</span>
                    </a>
                </div>
            </aside>
        `;

        if (window.lucide) {
            window.lucide.createIcons();
        }
    }
}

customElements.define('app-sidebar', Sidebar);

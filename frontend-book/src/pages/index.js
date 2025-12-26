import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import styles from './index.module.css';

function HomepageHeader() {
    const { siteConfig } = useDocusaurusContext();
    return (
        <header className={clsx('hero hero--primary', styles.heroBanner)}>
            <div className="container">
                <div className="row">
                    <div className="col col--6" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                        <Heading as="h1" className="hero__title">
                            {siteConfig.title}
                        </Heading>
                        <p className="hero__subtitle">{siteConfig.tagline}</p>
                        <div className={styles.buttons}>
                            <Link
                                className="button button--secondary button--lg"
                                to="/docs/intro">
                                START READING ➔
                            </Link>
                        </div>
                    </div>
                    <div className="col col--6">
                        <img src="img/hero_robot_hand.png" alt="Robot Hand" style={{ borderRadius: '10px', boxShadow: '0 0 30px rgba(0,210,255,0.3)' }} />
                    </div>
                </div>
            </div>
        </header>
    );
}

function Feature({ imageUrl, title, description, link }) {
    return (
        <div className={clsx('col col--3')}>
            <div className="feature-card">
                <div className="text--center">
                    {imageUrl && <img src={imageUrl} alt={title} style={{ height: '150px', objectFit: 'cover', width: '100%', borderRadius: '10px', marginBottom: '1rem' }} />}
                </div>
                <div className="text--center padding-horiz--md">
                    <Heading as="h3">{title}</Heading>
                    <p>{description}</p>
                    <Link to={link || "#"} className="button button--outline button--primary button--sm">Learn More</Link>
                </div>
            </div>
        </div>
    );
}

export default function Home() {
    const { siteConfig } = useDocusaurusContext();

    const pillars = [
        {
            title: 'The Nervous System',
            description: 'ROS 2 Middleware, Real-time Communication (DDS), and Node Graphs.',
            link: '/docs/module-1-ros2-nervous-system',
            // Using existing SVGs or just abstract for now if images run out
            imageUrl: 'img/ros2_nervous.png'
        },
        {
            title: 'The Body',
            description: 'Actuators, Sensors, Power Systems, and Kinematics.',
            link: '/docs/hardware-basics/module-3-hardware',
            imageUrl: 'img/hardware_chip.png'
        },
        {
            title: 'The Brain',
            description: 'Vision-Language-Action (VLA) Models, Transformers, and LLMs.',
            link: '/docs/vla-systems/module-4-vla-foundations',
            imageUrl: 'img/hero_robot_hand.png' // Reusing hero for Brain aesthetic
        },
        {
            title: 'The Digital Twin',
            description: 'Gazebo & Unity Simulation, URDF Modeling, and Sim-to-Real.',
            link: '/docs/module-2-digital-twin-simulation',
            imageUrl: 'img/simulation_split.png'
        },
    ];

    return (
        <Layout
            title={`Home`}
            description="The Textbook for Physical AI">
            <HomepageHeader />
            <main>
                <section style={{ padding: '4rem 0', background: 'var(--site-bg-dark)' }}>
                    <div className="container">
                        <h2 style={{ textAlign: 'center', marginBottom: '3rem', fontSize: '2.5rem', color: 'var(--ifm-color-primary)' }}>The 4 Pillars of Physical AI</h2>
                        <div className="row">
                            {pillars.map((props, idx) => (
                                <Feature key={idx} {...props} />
                            ))}
                        </div>
                    </div>
                </section>
            </main>
        </Layout>
    );
}

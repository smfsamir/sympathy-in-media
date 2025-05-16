import { useState, FormEvent } from 'react';
import { useRouter } from 'next/router';

const LoginForm: React.FC = () => {
    // Define state with TypeScript types
    const [email, setEmail] = useState<string>('');
    const [password, setPassword] = useState<string>('');
    const [message, setMessage] = useState<string>('');
    const router = useRouter()

    // Handle form submission
    const handleSubmit = async (e: FormEvent) => {
        e.preventDefault();

        try {
            const response = await fetch('/api/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password }),
            });

            const data = await response.json();

            if (response.ok) {
                setMessage(data.message);
                router.push({
                    pathname: '/dashboard',
                    query: { email: email, 
                    },
                })
            } else {
                setMessage('Failed to save credentials.');
            }
        } catch (error) {
            console.error('Error saving credentials:', error);
            setMessage('An error occurred.');
        }
    };

    return (
        <div
            style={{
                maxWidth: '400px',
                margin: 'auto',
                padding: '20px',
                border: '1px solid #ddd',
                borderRadius: '5px',
            }}
        >
            <h2>Login</h2>
            <form onSubmit={handleSubmit}>
                <div style={{ marginBottom: '15px' }}>
                    <label htmlFor="email" style={{ display: 'block', marginBottom: '5px' }}>
                        Email
                    </label>
                    <input
                        type="email"
                        id="email"
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                        required
                        style={{
                            width: '100%',
                            padding: '8px',
                            border: '1px solid #ccc',
                            borderRadius: '4px',
                        }}
                    />
                </div>
                <div style={{ marginBottom: '15px' }}>
                    <label htmlFor="password" style={{ display: 'block', marginBottom: '5px' }}>
                        Password
                    </label>
                    <input
                        type="password"
                        id="password"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        required
                        style={{
                            width: '100%',
                            padding: '8px',
                            border: '1px solid #ccc',
                            borderRadius: '4px',
                        }}
                    />
                </div>
                <button
                    type="submit"
                    style={{
                        width: '100%',
                        padding: '10px',
                        backgroundColor: '#0070f3',
                        color: '#fff',
                        border: 'none',
                        borderRadius: '4px',
                        cursor: 'pointer',
                    }}
                >
                    Login
                </button>
            </form>
            {message && (
                <p style={{ marginTop: '20px', color: 'green' }}>
                    {message}
                </p>
            )}
        </div>
    );
};

export default LoginForm;

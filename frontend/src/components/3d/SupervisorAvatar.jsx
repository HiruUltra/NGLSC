import React, { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import { Html } from '@react-three/drei';

/**
 * SupervisorAvatar Component
 * A professional supervisor with hair, sunglasses, and name label.
 */
const SupervisorAvatar = React.memo(({ position = [0, 0, 0], pathLength = 8, speed = 0.5, currentAlert }) => {
    const groupRef = useRef();
    const isAlerting = currentAlert && (currentAlert.severity === 'critical' || currentAlert.severity === 'warning');

    const materialColor = useMemo(() => {
        if (!isAlerting) return "#1a1a2e";
        return currentAlert.severity === 'critical' ? "#ff0000" : "#ff9f43";
    }, [isAlerting, currentAlert?.severity]);

    const skinColor = useMemo(() => {
        return isAlerting ? "#ff7675" : "#f3e5ab";
    }, [isAlerting]);

    useFrame((state) => {
        if (!groupRef.current) return;

        if (isAlerting) {
            groupRef.current.lookAt(0, groupRef.current.position.y, 0);
            groupRef.current.position.y = Math.abs(Math.sin(state.clock.getElapsedTime() * 10)) * 0.02;
            return;
        }

        const t = state.clock.getElapsedTime() * speed;
        const xOffset = Math.sin(t) * pathLength;

        groupRef.current.position.x = position[0] + xOffset;

        const direction = Math.cos(t);
        if (Math.abs(direction) > 0.1) {
            groupRef.current.rotation.y = direction > 0 ? Math.PI / 2 : -Math.PI / 2;
        }

        groupRef.current.position.y = Math.abs(Math.sin(t * 4)) * 0.05;
    });

    return (
        <group ref={groupRef} position={position}>
            {/* ── Name Label ── */}
            <Html
                position={[0, 2.05, 0]}
                center
                distanceFactor={6}
                style={{ pointerEvents: 'none' }}
            >
                <div style={{
                    background: 'linear-gradient(135deg, #e74c3c, #c0392b)',
                    color: '#fff',
                    padding: '3px 12px',
                    borderRadius: '8px',
                    fontSize: '11px',
                    fontWeight: 'bold',
                    letterSpacing: '1.5px',
                    textTransform: 'uppercase',
                    whiteSpace: 'nowrap',
                    textAlign: 'center',
                    border: '1px solid rgba(255,255,255,0.3)',
                    boxShadow: '0 2px 8px rgba(0,0,0,0.4)',
                }}>
                    🛡️ Supervisor
                </div>
            </Html>

            {/* ── Professional Torso (Suit) ── */}
            <mesh position={[0, 1.1, 0]} scale={[1, 1, 0.6]} castShadow>
                <capsuleGeometry args={[0.22, 0.45, 8, 8]} />
                <meshStandardMaterial color={materialColor} />
            </mesh>
            {/* Suit Lapel Detail */}
            <mesh position={[0, 1.25, 0.12]} castShadow>
                <boxGeometry args={[0.12, 0.2, 0.01]} />
                <meshStandardMaterial color="#ecf0f1" />
            </mesh>
            {/* Tie */}
            <mesh position={[0, 1.08, 0.12]} castShadow>
                <boxGeometry args={[0.05, 0.35, 0.01]} />
                <meshStandardMaterial color="#c0392b" />
            </mesh>
            {/* Tie Knot */}
            <mesh position={[0, 1.26, 0.13]} castShadow>
                <sphereGeometry args={[0.025, 6, 6]} />
                <meshStandardMaterial color="#c0392b" />
            </mesh>

            {/* ── Neck ── */}
            <mesh position={[0, 1.45, 0]} castShadow>
                <sphereGeometry args={[0.08, 8, 8]} />
                <meshStandardMaterial color={skinColor} />
            </mesh>

            {/* ── Head ── */}
            <group position={[0, 1.6, 0]}>
                <mesh castShadow>
                    <sphereGeometry args={[0.15, 16, 16]} />
                    <meshStandardMaterial color={skinColor} />
                </mesh>

                {/* ── Hair (slicked back professional style) ── */}
                {/* Top hair */}
                <mesh position={[0, 0.08, -0.03]} rotation={[-0.2, 0, 0]} castShadow>
                    <sphereGeometry args={[0.155, 16, 16, 0, Math.PI * 2, 0, Math.PI / 2]} />
                    <meshStandardMaterial color="#1a1a1a" roughness={0.6} />
                </mesh>
                {/* Side hair left */}
                <mesh position={[-0.13, 0.02, 0]} castShadow>
                    <boxGeometry args={[0.04, 0.12, 0.15]} />
                    <meshStandardMaterial color="#1a1a1a" roughness={0.6} />
                </mesh>
                {/* Side hair right */}
                <mesh position={[0.13, 0.02, 0]} castShadow>
                    <boxGeometry args={[0.04, 0.12, 0.15]} />
                    <meshStandardMaterial color="#1a1a1a" roughness={0.6} />
                </mesh>
                {/* Back hair */}
                <mesh position={[0, 0.02, -0.12]} castShadow>
                    <boxGeometry args={[0.22, 0.14, 0.06]} />
                    <meshStandardMaterial color="#1a1a1a" roughness={0.6} />
                </mesh>

                {/* ── Sunglasses ── */}
                {/* Left Lens */}
                <mesh position={[-0.055, 0.02, 0.14]}>
                    <boxGeometry args={[0.07, 0.04, 0.01]} />
                    <meshStandardMaterial color="#111111" metalness={0.9} roughness={0.1} />
                </mesh>
                {/* Right Lens */}
                <mesh position={[0.055, 0.02, 0.14]}>
                    <boxGeometry args={[0.07, 0.04, 0.01]} />
                    <meshStandardMaterial color="#111111" metalness={0.9} roughness={0.1} />
                </mesh>
                {/* Bridge */}
                <mesh position={[0, 0.02, 0.145]}>
                    <boxGeometry args={[0.03, 0.015, 0.005]} />
                    <meshStandardMaterial color="#333333" metalness={0.8} roughness={0.2} />
                </mesh>
                {/* Left Temple */}
                <mesh position={[-0.09, 0.02, 0.08]} rotation={[0, 0.4, 0]}>
                    <boxGeometry args={[0.005, 0.015, 0.12]} />
                    <meshStandardMaterial color="#333333" metalness={0.8} roughness={0.2} />
                </mesh>
                {/* Right Temple */}
                <mesh position={[0.09, 0.02, 0.08]} rotation={[0, -0.4, 0]}>
                    <boxGeometry args={[0.005, 0.015, 0.12]} />
                    <meshStandardMaterial color="#333333" metalness={0.8} roughness={0.2} />
                </mesh>

                {/* Nose */}
                <mesh position={[0, -0.03, 0.145]}>
                    <sphereGeometry args={[0.02, 6, 6]} />
                    <meshStandardMaterial color={skinColor} />
                </mesh>

                {/* Ears */}
                <mesh position={[0.15, 0, 0]}>
                    <sphereGeometry args={[0.025, 6, 6]} />
                    <meshStandardMaterial color={skinColor} />
                </mesh>
                <mesh position={[-0.15, 0, 0]}>
                    <sphereGeometry args={[0.025, 6, 6]} />
                    <meshStandardMaterial color={skinColor} />
                </mesh>
            </group>

            {/* ── Shoulder Joints ── */}
            <mesh position={[0.26, 1.25, 0]} castShadow>
                <sphereGeometry args={[0.07, 8, 8]} />
                <meshStandardMaterial color={materialColor} />
            </mesh>
            <mesh position={[-0.26, 1.25, 0]} castShadow>
                <sphereGeometry args={[0.07, 8, 8]} />
                <meshStandardMaterial color={materialColor} />
            </mesh>

            {/* ── Arms (suit sleeves) ── */}
            <mesh position={[0.26, 0.95, 0]} castShadow>
                <capsuleGeometry args={[0.05, 0.4, 4, 8]} />
                <meshStandardMaterial color={materialColor} />
            </mesh>
            <mesh position={[-0.26, 0.95, 0]} castShadow>
                <capsuleGeometry args={[0.05, 0.4, 4, 8]} />
                <meshStandardMaterial color={materialColor} />
            </mesh>

            {/* ── Hands ── */}
            <mesh position={[0.26, 0.72, 0]} castShadow>
                <sphereGeometry args={[0.05, 8, 8]} scale={[1, 1.2, 0.6]} />
                <meshStandardMaterial color={skinColor} />
            </mesh>
            <mesh position={[-0.26, 0.72, 0]} castShadow>
                <sphereGeometry args={[0.05, 8, 8]} scale={[1, 1.2, 0.6]} />
                <meshStandardMaterial color={skinColor} />
            </mesh>

            {/* ── Professional ID Badge ── */}
            <mesh position={[0.12, 1.15, 0.14]} castShadow>
                <boxGeometry args={[0.08, 0.1, 0.005]} />
                <meshStandardMaterial color="#f1c40f" />
            </mesh>
            {/* Badge clip */}
            <mesh position={[0.12, 1.21, 0.14]}>
                <boxGeometry args={[0.04, 0.02, 0.005]} />
                <meshStandardMaterial color="#bdc3c7" metalness={0.8} />
            </mesh>

            {/* ── Belt ── */}
            <mesh position={[0, 0.82, 0]} castShadow>
                <cylinderGeometry args={[0.19, 0.19, 0.04, 8]} />
                <meshStandardMaterial color="#2c3e50" />
            </mesh>
            {/* Belt Buckle */}
            <mesh position={[0, 0.82, 0.18]}>
                <boxGeometry args={[0.06, 0.04, 0.01]} />
                <meshStandardMaterial color="#f1c40f" metalness={0.9} roughness={0.1} />
            </mesh>

            {/* ── Hip Joints ── */}
            <mesh position={[0.13, 0.75, 0]} castShadow>
                <sphereGeometry args={[0.08, 8, 8]} />
                <meshStandardMaterial color="#1a252f" />
            </mesh>
            <mesh position={[-0.13, 0.75, 0]} castShadow>
                <sphereGeometry args={[0.08, 8, 8]} />
                <meshStandardMaterial color="#1a252f" />
            </mesh>

            {/* ── Legs (formal trousers) ── */}
            <mesh position={[0.13, 0.4, 0]} castShadow>
                <capsuleGeometry args={[0.065, 0.6, 4, 8]} />
                <meshStandardMaterial color="#1a252f" />
            </mesh>
            <mesh position={[-0.13, 0.4, 0]} castShadow>
                <capsuleGeometry args={[0.065, 0.6, 4, 8]} />
                <meshStandardMaterial color="#1a252f" />
            </mesh>

            {/* ── Shoes ── */}
            <mesh position={[0.13, 0.06, 0.04]} castShadow>
                <boxGeometry args={[0.08, 0.06, 0.16]} />
                <meshStandardMaterial color="#111111" />
            </mesh>
            <mesh position={[-0.13, 0.06, 0.04]} castShadow>
                <boxGeometry args={[0.08, 0.06, 0.16]} />
                <meshStandardMaterial color="#111111" />
            </mesh>
        </group>
    );
});

export default SupervisorAvatar;

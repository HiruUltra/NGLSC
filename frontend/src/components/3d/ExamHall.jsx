import React, { Suspense, useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Html, RoundedBox, Cylinder } from '@react-three/drei';
import StudentAvatar from './StudentAvatar';
import SupervisorAvatar from './SupervisorAvatar';

// ─── Memoized Sub-Components ───────────────────────────────────────

const Clock = React.memo(({ position }) => (
    <group position={position}>
        {/* Metallic Rim */}
        <mesh castShadow>
            <cylinderGeometry args={[0.55, 0.55, 0.08, 16]} />
            <meshStandardMaterial color="#bdc3c7" metalness={0.8} roughness={0.2} />
        </mesh>
        {/* Dial Face */}
        <mesh position={[0, 0, 0.04]} castShadow>
            <cylinderGeometry args={[0.5, 0.5, 0.02, 16]} />
            <meshStandardMaterial color="#ffffff" roughness={0.1} />
        </mesh>
        {/* Glass Cover — replaced meshPhysicalMaterial with cheap transparent standard */}
        <mesh position={[0, 0, 0.06]}>
            <cylinderGeometry args={[0.5, 0.5, 0.01, 16]} />
            <meshStandardMaterial color="#cde8ff" transparent opacity={0.25} roughness={0.1} />
        </mesh>
        {/* Clock Hands */}
        <mesh position={[0, 0, 0.05]}>
            <boxGeometry args={[0.02, 0.3, 0.01]} />
            <meshStandardMaterial color="#2c3e50" />
        </mesh>
        <mesh position={[0.1, 0, 0.05]} rotation={[0, 0, -Math.PI / 3]}>
            <boxGeometry args={[0.02, 0.2, 0.01]} />
            <meshStandardMaterial color="#2c3e50" />
        </mesh>
    </group>
));

// Pure 3D poster — no Html/DOM overlay
const Poster = React.memo(({ position, rotation, color = "#3498db", title = "EXAM RULES" }) => (
    <group position={position} rotation={rotation}>
        <mesh receiveShadow>
            <boxGeometry args={[1.5, 2, 0.02]} />
            <meshStandardMaterial color="#ffffff" roughness={0.5} />
        </mesh>
        {/* Colored header bar */}
        <mesh position={[0, 0.7, 0.02]}>
            <planeGeometry args={[1.3, 0.2]} />
            <meshStandardMaterial color={color} />
        </mesh>
        {/* Simulated text lines */}
        {[0.3, 0.1, -0.1, -0.3].map((y, i) => (
            <mesh key={i} position={[0, y, 0.02]}>
                <planeGeometry args={[1.1, 0.06]} />
                <meshStandardMaterial color="#dfe6e9" />
            </mesh>
        ))}
    </group>
));

const Laptop = React.memo(({ position, rotation = [0, 0, 0] }) => (
    <group position={position} rotation={rotation}>
        {/* Base */}
        <RoundedBox args={[0.4, 0.02, 0.25]} radius={0.01} smoothness={2} position={[0, 0.01, 0]} castShadow>
            <meshStandardMaterial color="#1e272e" metalness={0.6} roughness={0.4} />
        </RoundedBox>
        {/* Screen (hinged open) */}
        <group position={[0, 0.02, -0.12]} rotation={[Math.PI / 6, 0, 0]}>
            <RoundedBox args={[0.4, 0.25, 0.02]} radius={0.01} smoothness={2} position={[0, 0.125, 0]} castShadow>
                <meshStandardMaterial color="#1e272e" metalness={0.6} roughness={0.4} />
            </RoundedBox>
            {/* Emissive Screen Area */}
            <mesh position={[0, 0.125, 0.011]}>
                <planeGeometry args={[0.36, 0.21]} />
                <meshStandardMaterial color="#ffffff" emissive="#3498db" emissiveIntensity={0.5} />
            </mesh>
        </group>
    </group>
));

const Desk = React.memo(({ position }) => (
    <group position={position}>
        {/* Modern Beveled Table Top */}
        <RoundedBox args={[1.2, 0.04, 0.8]} radius={0.02} smoothness={2} position={[0, 0.75, 0]} castShadow receiveShadow>
            <meshStandardMaterial color="#ecf0f1" roughness={0.2} metalness={0.1} />
        </RoundedBox>

        {/* Modern "Sled" Frame Legs */}
        <group position={[0.55, 0.375, 0]}>
            <Cylinder args={[0.02, 0.02, 0.75, 8]} position={[0, 0, 0.35]} castShadow>
                <meshStandardMaterial color="#2d3436" metalness={0.8} roughness={0.2} />
            </Cylinder>
            <Cylinder args={[0.02, 0.02, 0.75, 8]} position={[0, 0, -0.35]} castShadow>
                <meshStandardMaterial color="#2d3436" metalness={0.8} roughness={0.2} />
            </Cylinder>
            <Cylinder args={[0.02, 0.02, 0.75, 8]} rotation={[Math.PI / 2, 0, 0]} position={[0, -0.37, 0]} castShadow>
                <meshStandardMaterial color="#2d3436" metalness={0.8} roughness={0.2} />
            </Cylinder>
        </group>
        <group position={[-0.55, 0.375, 0]}>
            <Cylinder args={[0.02, 0.02, 0.75, 8]} position={[0, 0, 0.35]} castShadow>
                <meshStandardMaterial color="#2d3436" metalness={0.8} roughness={0.2} />
            </Cylinder>
            <Cylinder args={[0.02, 0.02, 0.75, 8]} position={[0, 0, -0.35]} castShadow>
                <meshStandardMaterial color="#2d3436" metalness={0.8} roughness={0.2} />
            </Cylinder>
            <Cylinder args={[0.02, 0.02, 0.75, 8]} rotation={[Math.PI / 2, 0, 0]} position={[0, -0.37, 0]} castShadow>
                <meshStandardMaterial color="#2d3436" metalness={0.8} roughness={0.2} />
            </Cylinder>
        </group>

        {/* Laptop replacing flat plane */}
        <Laptop position={[0, 0.77, 0]} />
    </group>
));

const Chair = React.memo(({ position, rotation = [0, 0, 0] }) => (
    <group position={position} rotation={rotation}>
        {/* Ergonomic Curved Seat */}
        <RoundedBox args={[0.45, 0.08, 0.45]} radius={0.04} smoothness={2} position={[0, 0.45, 0]} castShadow>
            <meshStandardMaterial color="#34495e" roughness={0.8} />
        </RoundedBox>

        {/* Ergonomic Curved Backrest */}
        <RoundedBox args={[0.45, 0.4, 0.05]} radius={0.02} smoothness={2} position={[0, 0.75, -0.2]} rotation={[0.1, 0, 0]} castShadow>
            <meshStandardMaterial color="#34495e" roughness={0.8} />
        </RoundedBox>

        {/* Armature connecting backrest to seat */}
        <mesh position={[0, 0.55, -0.2]} castShadow>
            <boxGeometry args={[0.1, 0.2, 0.02]} />
            <meshStandardMaterial color="#2c3e50" metalness={0.8} roughness={0.2} />
        </mesh>

        {/* Central Metal Pole */}
        <Cylinder args={[0.03, 0.03, 0.4, 8]} position={[0, 0.2, 0]} castShadow>
            <meshStandardMaterial color="#bdc3c7" metalness={0.9} roughness={0.1} />
        </Cylinder>

        {/* 5-Point Star Base with Casters */}
        {[0, 1, 2, 3, 4].map((i) => {
            const angle = (i * Math.PI * 2) / 5;
            return (
                <group key={i} rotation={[0, angle, 0]}>
                    <mesh position={[0, 0.05, 0.15]} rotation={[Math.PI / 2, 0, 0]} castShadow>
                        <cylinderGeometry args={[0.02, 0.02, 0.3, 8]} />
                        <meshStandardMaterial color="#7f8c8d" metalness={0.6} roughness={0.4} />
                    </mesh>
                    {/* Caster Wheel */}
                    <mesh position={[0, 0.03, 0.3]} castShadow>
                        <sphereGeometry args={[0.03, 8, 8]} />
                        <meshStandardMaterial color="#2c3e50" />
                    </mesh>
                </group>
            );
        })}
    </group>
));

// Helper: Generate consistent deterministic variations based on position
const getNPCStyle = (x, z) => {
    const seed = Math.abs(x * 13 + z * 7);
    const hairColors = ['#1a1a1a', '#3b2f2f', '#4a3b32', '#2c1e16', '#d6b268'];
    const shirtColors = ['#e74c3c', '#2ecc71', '#9b59b6', '#34495e', '#16a085', '#27ae60', '#8e44ad'];
    return {
        hairColor: hairColors[seed % hairColors.length],
        shirtColor: shirtColors[seed % shirtColors.length],
        hairStyle: (seed % 3) + 1
    };
};

// Deterministic occupancy — replaces Math.random() so it never changes on re-render
const isNPCSeatOccupied = (x, z) => ((x * 17 + z * 31) % 10) > 3;

// NPC student names for occupied seats
const NPC_NAMES = [
    'Amal K.', 'Sithara D.', 'Kavinda R.', 'Dilini P.', 'Nuwan S.',
    'Rashmika T.', 'Tharindu M.', 'Sachini F.', 'Isuru W.', 'Nethmi B.',
    'Chamara J.', 'Hiruni L.', 'Dinesh A.', 'Sanduni G.', 'Kavindu N.'
];
const getNPCName = (x, z) => NPC_NAMES[Math.abs(x * 7 + z * 11) % NPC_NAMES.length];

// Seat number label component
const SeatLabel = React.memo(({ position, seatNumber }) => (
    <Html position={position} center distanceFactor={8} style={{ pointerEvents: 'none' }}>
        <div style={{
            background: 'linear-gradient(135deg, #2d3436, #636e72)',
            color: '#fff',
            padding: '2px 8px',
            borderRadius: '6px',
            fontSize: '10px',
            fontWeight: 'bold',
            whiteSpace: 'nowrap',
            textAlign: 'center',
            border: '1px solid rgba(255,255,255,0.2)',
            boxShadow: '0 2px 6px rgba(0,0,0,0.4)',
        }}>
            Seat {seatNumber}
        </div>
    </Html>
));

// Student name label
const NameLabel = React.memo(({ position, name, isMainUser = false }) => (
    <Html position={position} center distanceFactor={6} style={{ pointerEvents: 'none' }}>
        <div style={{
            background: isMainUser
                ? 'linear-gradient(135deg, #00b894, #00cec9)'
                : 'rgba(45, 52, 54, 0.85)',
            color: '#fff',
            padding: '2px 10px',
            borderRadius: '8px',
            fontSize: '10px',
            fontWeight: 'bold',
            whiteSpace: 'nowrap',
            textAlign: 'center',
            border: isMainUser ? '1px solid rgba(255,255,255,0.4)' : '1px solid rgba(255,255,255,0.15)',
            boxShadow: isMainUser ? '0 2px 10px rgba(0,206,201,0.4)' : '0 1px 4px rgba(0,0,0,0.3)',
            letterSpacing: '0.5px',
        }}>
            {isMainUser ? '👤 ' : ''}{name}
        </div>
    </Html>
));

const OtherStudentAvatar = React.memo(({ position, style, rotation = [0, 0, 0] }) => {
    const headTilt = Math.sin(position[0] * position[2]) * 0.1;

    return (
        <group position={position} rotation={rotation}>
            {/* Organic Body/Torso */}
            <mesh position={[0, 0.55, 0]} scale={[1, 1, 0.6]} castShadow>
                <capsuleGeometry args={[0.16, 0.3, 8, 8]} />
                <meshStandardMaterial color={style.shirtColor} />
            </mesh>

            {/* Organic Shoulder Joints */}
            <mesh position={[0.2, 0.65, 0]} castShadow>
                <sphereGeometry args={[0.07, 8, 8]} />
                <meshStandardMaterial color={style.shirtColor} />
            </mesh>
            <mesh position={[-0.2, 0.65, 0]} castShadow>
                <sphereGeometry args={[0.07, 8, 8]} />
                <meshStandardMaterial color={style.shirtColor} />
            </mesh>

            {/* Head (tilted slightly as if looking at paper) */}
            <group position={[0, 0.98, 0]} rotation={[0.1 + headTilt, 0, 0]}>
                <mesh castShadow>
                    <sphereGeometry args={[0.12, 16, 16]} />
                    <meshStandardMaterial color="#f3e5ab" />
                </mesh>

                {/* Eyes */}
                <mesh position={[0.045, 0.01, 0.115]}>
                    <sphereGeometry args={[0.015, 6, 6]} />
                    <meshStandardMaterial color="#1a1a1a" roughness={0.5} />
                </mesh>
                <mesh position={[-0.045, 0.01, 0.115]}>
                    <sphereGeometry args={[0.015, 6, 6]} />
                    <meshStandardMaterial color="#1a1a1a" roughness={0.5} />
                </mesh>

                {/* Nose */}
                <mesh position={[0, -0.03, 0.125]}>
                    <sphereGeometry args={[0.015, 6, 6]} />
                    <meshStandardMaterial color="#e3cba8" />
                </mesh>

                {/* Ears */}
                <mesh position={[0.125, 0, 0]}>
                    <sphereGeometry args={[0.02, 6, 6]} />
                    <meshStandardMaterial color="#f3e5ab" />
                </mesh>
                <mesh position={[-0.125, 0, 0]}>
                    <sphereGeometry args={[0.02, 6, 6]} />
                    <meshStandardMaterial color="#f3e5ab" />
                </mesh>

                {/* Varied Hair */}
                {style.hairStyle === 1 && (
                    <group>
                        <mesh position={[0, 0.05, -0.02]} rotation={[-0.2, 0, 0]} castShadow>
                            <sphereGeometry args={[0.13, 16, 16, 0, Math.PI * 2, 0, Math.PI / 2]} />
                            <meshStandardMaterial color={style.hairColor} roughness={0.8} />
                        </mesh>
                        <mesh position={[0, 0.1, 0.08]} rotation={[0.2, 0, 0]} castShadow>
                            <boxGeometry args={[0.2, 0.05, 0.1]} />
                            <meshStandardMaterial color={style.hairColor} roughness={0.8} />
                        </mesh>
                    </group>
                )}
                {style.hairStyle === 2 && (
                    <group>
                        <mesh position={[0, 0.04, 0]} rotation={[-0.1, 0, 0]} castShadow>
                            <sphereGeometry args={[0.125, 16, 16, 0, Math.PI * 2, 0, Math.PI / 2]} />
                            <meshStandardMaterial color={style.hairColor} roughness={0.9} />
                        </mesh>
                        <mesh position={[0, 0.12, -0.02]} castShadow>
                            <boxGeometry args={[0.15, 0.06, 0.15]} />
                            <meshStandardMaterial color={style.hairColor} roughness={0.9} />
                        </mesh>
                    </group>
                )}
                {style.hairStyle === 3 && (
                    <group>
                        <mesh position={[0, 0.08, -0.04]} rotation={[-0.3, 0, 0]} castShadow>
                            <sphereGeometry args={[0.135, 16, 16, 0, Math.PI * 2, 0, Math.PI / 1.8]} />
                            <meshStandardMaterial color={style.hairColor} roughness={0.7} />
                        </mesh>
                        <mesh position={[0.05, 0.05, 0.08]} rotation={[0.1, 0.2, 0]} castShadow>
                            <boxGeometry args={[0.1, 0.1, 0.1]} />
                            <meshStandardMaterial color={style.hairColor} roughness={0.7} />
                        </mesh>
                        <mesh position={[-0.05, 0.05, 0.08]} rotation={[0.1, -0.2, 0]} castShadow>
                            <boxGeometry args={[0.1, 0.1, 0.1]} />
                            <meshStandardMaterial color={style.hairColor} roughness={0.7} />
                        </mesh>
                    </group>
                )}
            </group>

            {/* Right Arm - reaching forward onto desk */}
            <group position={[0.18, 0.58, 0]}>
                <mesh position={[0, -0.05, 0.1]} rotation={[-0.6, 0, 0]} castShadow>
                    <capsuleGeometry args={[0.04, 0.22, 4, 6]} />
                    <meshStandardMaterial color={style.shirtColor} />
                </mesh>
                <mesh position={[0, -0.1, 0.3]} rotation={[-1.5, 0, 0]} castShadow>
                    <capsuleGeometry args={[0.035, 0.18, 4, 6]} />
                    <meshStandardMaterial color={style.shirtColor} />
                </mesh>
                <group position={[0, -0.1, 0.48]}>
                    <mesh scale={[1, 0.4, 1.4]} castShadow>
                        <sphereGeometry args={[0.04, 8, 8]} />
                        <meshStandardMaterial color="#f3e5ab" />
                    </mesh>
                    <mesh position={[0, 0, 0.05]} rotation={[-1.5, 0, 0]} castShadow>
                        <capsuleGeometry args={[0.012, 0.04, 3, 4]} />
                        <meshStandardMaterial color="#f3e5ab" />
                    </mesh>
                    <mesh position={[0.025, 0, 0.045]} rotation={[-1.5, 0, -0.15]} castShadow>
                        <capsuleGeometry args={[0.01, 0.035, 3, 4]} />
                        <meshStandardMaterial color="#f3e5ab" />
                    </mesh>
                    <mesh position={[-0.025, 0, 0.045]} rotation={[-1.5, 0, 0.15]} castShadow>
                        <capsuleGeometry args={[0.01, 0.035, 3, 4]} />
                        <meshStandardMaterial color="#f3e5ab" />
                    </mesh>
                </group>
            </group>

            {/* Left Arm - reaching forward onto desk */}
            <group position={[-0.18, 0.58, 0]}>
                <mesh position={[0, -0.05, 0.1]} rotation={[-0.6, 0, 0]} castShadow>
                    <capsuleGeometry args={[0.04, 0.22, 4, 6]} />
                    <meshStandardMaterial color={style.shirtColor} />
                </mesh>
                <mesh position={[0, -0.1, 0.3]} rotation={[-1.5, 0, 0]} castShadow>
                    <capsuleGeometry args={[0.035, 0.18, 4, 6]} />
                    <meshStandardMaterial color={style.shirtColor} />
                </mesh>
                <group position={[0, -0.1, 0.48]}>
                    <mesh scale={[1, 0.4, 1.4]} castShadow>
                        <sphereGeometry args={[0.04, 8, 8]} />
                        <meshStandardMaterial color="#f3e5ab" />
                    </mesh>
                    <mesh position={[0, 0, 0.05]} rotation={[-1.5, 0, 0]} castShadow>
                        <capsuleGeometry args={[0.012, 0.04, 3, 4]} />
                        <meshStandardMaterial color="#f3e5ab" />
                    </mesh>
                    <mesh position={[0.025, 0, 0.045]} rotation={[-1.5, 0, -0.15]} castShadow>
                        <capsuleGeometry args={[0.01, 0.035, 3, 4]} />
                        <meshStandardMaterial color="#f3e5ab" />
                    </mesh>
                    <mesh position={[-0.025, 0, 0.045]} rotation={[-1.5, 0, 0.15]} castShadow>
                        <capsuleGeometry args={[0.01, 0.035, 3, 4]} />
                        <meshStandardMaterial color="#f3e5ab" />
                    </mesh>
                </group>
            </group>

            {/* Legs */}
            <mesh position={[0.1, 0.25, 0.2]} rotation={[Math.PI / 2, 0, 0]}>
                <capsuleGeometry args={[0.05, 0.4, 4, 6]} />
                <meshStandardMaterial color="#2c3e50" />
            </mesh>
            <mesh position={[-0.1, 0.25, 0.2]} rotation={[Math.PI / 2, 0, 0]}>
                <capsuleGeometry args={[0.05, 0.4, 4, 6]} />
                <meshStandardMaterial color="#2c3e50" />
            </mesh>
            <mesh position={[0.1, 0.0, 0.4]} rotation={[0, 0, 0]}>
                <capsuleGeometry args={[0.04, 0.45, 4, 6]} />
                <meshStandardMaterial color="#2c3e50" />
            </mesh>
            <mesh position={[-0.1, 0.0, 0.4]} rotation={[0, 0, 0]}>
                <capsuleGeometry args={[0.04, 0.45, 4, 6]} />
                <meshStandardMaterial color="#2c3e50" />
            </mesh>
        </group>
    );
});

const Room = React.memo(({ currentAlert, userName }) => {
    const isCritical = currentAlert?.severity === 'critical';
    const isWarning = currentAlert?.severity === 'warning';

    const wallColor = "#2d2e36";
    const floorColor = "#22232a";
    const neonBlue = "#00f3ff";
    const neonRed = "#ff0055";

    const neonActiveColor = isCritical ? neonRed : (isWarning ? "#ffa502" : neonBlue);
    const neonIntensity = isCritical || isWarning ? 8 : 4;

    // Pre-compute desk layout with seat numbers and names
    const deskLayout = useMemo(() => {
        let seatNum = 0;
        return [0, 2, 4, 6].flatMap(x =>
            [-6, -3, 0, 3].map(z => {
                seatNum++;
                const isMainUser = (x === 4 && z === 0);
                return {
                    x, z,
                    seatNumber: seatNum,
                    isMainUser,
                    isOccupied: !isMainUser && isNPCSeatOccupied(x, z),
                    style: getNPCStyle(x, z),
                    npcName: getNPCName(x, z)
                };
            })
        );
    }, []);

    return (
        <group>
            {/* Floor */}
            <mesh rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
                <planeGeometry args={[30, 30]} />
                <meshStandardMaterial color={floorColor} roughness={0.7} metalness={0.2} />
            </mesh>

            {/* Front Wall (Whiteboard & Teacher Area) */}
            <group position={[0, 5, -10]}>
                <mesh receiveShadow>
                    <boxGeometry args={[20, 10, 0.1]} />
                    <meshStandardMaterial color={wallColor} roughness={0.9} />
                </mesh>

                {/* Modern Framed Whiteboard */}
                <group position={[0, 0, 0.1]}>
                    <RoundedBox args={[8.2, 4.2, 0.1]} radius={0.05} smoothness={2} castShadow receiveShadow>
                        <meshStandardMaterial color="#2d3436" metalness={0.8} roughness={0.2} />
                    </RoundedBox>
                    <mesh position={[0, 0, 0.06]} receiveShadow>
                        <planeGeometry args={[8, 4]} />
                        <meshStandardMaterial color="#ffffff" roughness={0.2} metalness={0.1} />
                    </mesh>
                    <mesh position={[0, -2.1, 0.1]} castShadow>
                        <boxGeometry args={[8.2, 0.05, 0.2]} />
                        <meshStandardMaterial color="#636e72" metalness={0.6} roughness={0.4} />
                    </mesh>
                </group>

                <mesh position={[0, -4.9, 0.1]} castShadow>
                    <boxGeometry args={[20, 0.2, 0.1]} />
                    <meshStandardMaterial color="#2d3436" />
                </mesh>

                <spotLight
                    position={[0, 4, 3]}
                    angle={0.7}
                    penumbra={1}
                    intensity={5}
                    color="#ffffff"
                    target-position={[0, 0, 0]}
                    castShadow
                />
            </group>

            {/* Back Wall (Door) */}
            <group position={[0, 5, 10]}>
                <mesh receiveShadow>
                    <boxGeometry args={[20, 10, 0.1]} />
                    <meshStandardMaterial color={wallColor} roughness={0.9} />
                </mesh>
                <group position={[4, -2.5, -0.1]}>
                    <mesh position={[0, 0, 0]}>
                        <boxGeometry args={[1.6, 4.1, 0.1]} />
                        <meshStandardMaterial color="#2c3e50" />
                    </mesh>
                    <mesh position={[0, 0, -0.05]}>
                        <boxGeometry args={[1.5, 4, 0.05]} />
                        <meshStandardMaterial color="#34495e" roughness={0.6} metalness={0.2} />
                    </mesh>
                    <mesh position={[0.6, 0, -0.1]}>
                        <cylinderGeometry args={[0.03, 0.03, 0.2, 8]} rotation={[0, 0, Math.PI / 2]} />
                        <meshStandardMaterial color="#bdc3c7" metalness={0.9} roughness={0.1} />
                    </mesh>
                </group>
                <mesh position={[0, -4.9, -0.1]} castShadow>
                    <boxGeometry args={[20, 0.2, 0.1]} />
                    <meshStandardMaterial color="#2d3436" />
                </mesh>
            </group>

            {/* Left Wall (Windows) */}
            <group position={[-10, 5, 0]} rotation={[0, Math.PI / 2, 0]}>
                <mesh receiveShadow>
                    <boxGeometry args={[20, 10, 0.1]} />
                    <meshStandardMaterial color={wallColor} roughness={0.9} />
                </mesh>
                <Poster position={[-4, 1.5, 0.1]} title="CALCULUS" color="#e67e22" />
                <Poster position={[4, 1.5, 0.1]} title="BIOLOGY" color="#2ecc71" />

                {[-6, 0, 6].map((z, i) => (
                    <group key={i} position={[z, 1, 0.15]}>
                        {/* Outside Scenery */}
                        <mesh position={[0, 0, -0.1]}>
                            <planeGeometry args={[2.4, 3.4]} />
                            <meshBasicMaterial color="#0A1128" />
                        </mesh>
                        {/* Window Glass — cheap transparent instead of meshPhysicalMaterial */}
                        <mesh>
                            <boxGeometry args={[2.5, 3.5, 0.05]} />
                            <meshStandardMaterial color="#a8d8ea" transparent opacity={0.2} roughness={0.1} />
                        </mesh>
                        {/* Outer Frame */}
                        <mesh>
                            <boxGeometry args={[2.6, 3.6, 0.1]} />
                            <meshStandardMaterial color="#bdc3c7" metalness={0.4} />
                        </mesh>
                        {/* Vertical Mullion */}
                        <mesh position={[0, 0, 0.01]}>
                            <boxGeometry args={[0.1, 3.5, 0.1]} />
                            <meshStandardMaterial color="#bdc3c7" metalness={0.4} />
                        </mesh>
                        {/* Horizontal Transom */}
                        <mesh position={[0, 0.5, 0.01]}>
                            <boxGeometry args={[2.5, 0.1, 0.1]} />
                            <meshStandardMaterial color="#bdc3c7" metalness={0.4} />
                        </mesh>
                    </group>
                ))}
                <mesh position={[0, -4.9, 0.1]} castShadow>
                    <boxGeometry args={[20, 0.2, 0.1]} />
                    <meshStandardMaterial color="#2d3436" />
                </mesh>
            </group>

            {/* Right Wall */}
            <group position={[10, 5, 0]} rotation={[0, -Math.PI / 2, 0]}>
                <mesh receiveShadow>
                    <boxGeometry args={[20, 10, 0.1]} />
                    <meshStandardMaterial color={wallColor} roughness={0.9} />
                </mesh>
                <Poster position={[-3, 1.5, 0.1]} title="EXAM CONDUCT" color="#c0392b" />
                <Poster position={[3, 1.5, 0.1]} title="TIME MANAGEMENT" color="#9b59b6" />
                <mesh position={[0, -4.9, 0.1]} castShadow>
                    <boxGeometry args={[20, 0.2, 0.1]} />
                    <meshStandardMaterial color="#2d3436" />
                </mesh>
            </group>

            <Clock position={[0, 7.5, -9.9]} />

            {/* Cinematic Neon Light Bars */}
            <group position={[0, 9.5, 0]}>
                <group position={[-9.8, -1.5, 0]}>
                    <mesh rotation={[Math.PI / 2, 0, 0]}>
                        <cylinderGeometry args={[0.08, 0.08, 16, 8]} />
                        <meshStandardMaterial
                            color="#ffffff"
                            emissive={neonActiveColor}
                            emissiveIntensity={neonIntensity}
                        />
                    </mesh>
                    <pointLight
                        position={[1, 0, 0]}
                        intensity={neonIntensity * 2}
                        distance={20}
                        color={neonActiveColor}
                    />
                </group>
                <group position={[9.8, -1.5, 0]}>
                    <mesh rotation={[Math.PI / 2, 0, 0]}>
                        <cylinderGeometry args={[0.08, 0.08, 16, 8]} />
                        <meshStandardMaterial
                            color="#ffffff"
                            emissive={neonActiveColor}
                            emissiveIntensity={neonIntensity}
                        />
                    </mesh>
                    <pointLight
                        position={[-1, 0, 0]}
                        intensity={neonIntensity * 2}
                        distance={20}
                        color={neonActiveColor}
                    />
                </group>
            </group>

            {/* Populate Desks, Chairs, and NPCs */}
            {deskLayout.map(({ x, z, seatNumber, isMainUser, isOccupied, style, npcName }) => (
                <group key={`${x}-${z}`}>
                    <Desk position={[x, 0, z]} />
                    <mesh position={[x, 0.78, z - 0.1]} rotation={[-Math.PI / 2, 0, 0]}>
                        <planeGeometry args={[0.4, 0.3]} />
                        <meshStandardMaterial color="#ffffff" roughness={0.9} />
                    </mesh>

                    {/* Seat number on front of desk */}
                    <SeatLabel position={[x, 0.6, z + 0.42]} seatNumber={seatNumber} />

                    <Chair position={[x, 0, z + 0.5]} rotation={[0, Math.PI, 0]} />

                    {/* NPC students with name labels */}
                    {isOccupied && (
                        <>
                            <OtherStudentAvatar position={[x, 0.25, z + 0.45]} style={style} rotation={[0, Math.PI, 0]} />
                            <NameLabel position={[x, 1.55, z + 0.45]} name={npcName} />
                        </>
                    )}
                </group>
            ))}
        </group>
    );
});

export default function ExamHall({ currentAlert, userName }) {
    const displayName = userName || 'You';

    return (
        <div className="w-full h-full bg-black overflow-hidden rounded-2xl border border-gray-800 relative">
            <Canvas shadows gl={{ antialias: false, powerPreference: 'high-performance' }}>
                <PerspectiveCamera makeDefault position={[0, 4, 7]} fov={50} />
                <OrbitControls enablePan={false} minPolarAngle={Math.PI / 8} maxPolarAngle={Math.PI / 2} minDistance={3} maxDistance={15} />

                {/* Cinematic Dim Lighting */}
                <ambientLight intensity={0.7} />
                <directionalLight position={[10, 15, 10]} intensity={1.8} color="#90cdf4" castShadow shadow-mapSize={[1024, 1024]} />
                <pointLight position={[0, 5, 2]} intensity={1.5} distance={20} color="#00f3ff" />

                <Suspense fallback={<Html center><div className="text-cyan-400 font-bold">Loading Exam Hall...</div></Html>}>
                    <Room currentAlert={currentAlert} userName={displayName} />
                    <SupervisorAvatar position={[2, 0, -4]} pathLength={8} speed={0.3} currentAlert={currentAlert} />

                    {/* Main User Student */}
                    <group position={[4, 0.25, 0.45]}>
                        <StudentAvatar position={[0, 0, 0]} hairStyle={1} rotation={[0, Math.PI, 0]} />
                        <NameLabel position={[0, 1.3, 0]} name={displayName} isMainUser />
                    </group>
                </Suspense>
            </Canvas>
        </div>
    );
}

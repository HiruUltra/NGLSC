import React from 'react';

const StudentAvatar = ({ position, rotation = [0, 0, 0], hairColor = "#3b2f2f", shirtColor = "#3498db", hairStyle = 1 }) => {
    return (
        <group position={position} rotation={rotation}>
            {/* Organic Body/Torso */}
            <mesh position={[0, 0.55, 0]} scale={[1, 1, 0.6]} castShadow>
                <capsuleGeometry args={[0.16, 0.3, 16, 16]} />
                <meshStandardMaterial color={shirtColor} />
            </mesh>

            {/* Organic Shoulder Joints */}
            <mesh position={[0.2, 0.65, 0]} castShadow>
                <sphereGeometry args={[0.07, 16, 16]} />
                <meshStandardMaterial color={shirtColor} />
            </mesh>
            <mesh position={[-0.2, 0.65, 0]} castShadow>
                <sphereGeometry args={[0.07, 16, 16]} />
                <meshStandardMaterial color={shirtColor} />
            </mesh>

            {/* Head */}
            <group position={[0, 0.98, 0]}>
                <mesh castShadow>
                    <sphereGeometry args={[0.12, 32, 32]} />
                    <meshStandardMaterial color="#f3e5ab" />
                </mesh>

                {/* Facial Features */}
                {/* Eyes */}
                <mesh position={[0.045, 0.02, 0.115]}>
                    <sphereGeometry args={[0.015, 16, 16]} />
                    <meshStandardMaterial color="#1a1a1a" roughness={0.5} />
                </mesh>
                <mesh position={[-0.045, 0.02, 0.115]}>
                    <sphereGeometry args={[0.015, 16, 16]} />
                    <meshStandardMaterial color="#1a1a1a" roughness={0.5} />
                </mesh>

                {/* Nose */}
                <mesh position={[0, -0.02, 0.125]}>
                    <sphereGeometry args={[0.015, 16, 16]} />
                    <meshStandardMaterial color="#e3cba8" />
                </mesh>

                {/* Ears */}
                <mesh position={[0.125, 0, 0]}>
                    <sphereGeometry args={[0.02, 16, 16]} />
                    <meshStandardMaterial color="#f3e5ab" />
                </mesh>
                <mesh position={[-0.125, 0, 0]}>
                    <sphereGeometry args={[0.02, 16, 16]} />
                    <meshStandardMaterial color="#f3e5ab" />
                </mesh>

                {/* Hair Variations */}
                {hairStyle === 1 && (
                    <group>
                        {/* Base + Bangs */}
                        <mesh position={[0, 0.05, -0.02]} rotation={[-0.2, 0, 0]} castShadow>
                            <sphereGeometry args={[0.13, 32, 32, 0, Math.PI * 2, 0, Math.PI / 2]} />
                            <meshStandardMaterial color={hairColor} roughness={0.8} />
                        </mesh>
                        <mesh position={[0, 0.1, 0.08]} rotation={[0.2, 0, 0]} castShadow>
                            <boxGeometry args={[0.2, 0.05, 0.1]} />
                            <meshStandardMaterial color={hairColor} roughness={0.8} />
                        </mesh>
                    </group>
                )}

                {hairStyle === 2 && (
                    <group>
                        {/* Short Buzzcut Base */}
                        <mesh position={[0, 0.04, 0]} rotation={[-0.1, 0, 0]} castShadow>
                            <sphereGeometry args={[0.125, 32, 32, 0, Math.PI * 2, 0, Math.PI / 2]} />
                            <meshStandardMaterial color={hairColor} roughness={0.9} />
                        </mesh>
                        {/* Small Spiky Top */}
                        <mesh position={[0, 0.12, -0.02]} castShadow>
                            <boxGeometry args={[0.15, 0.06, 0.15]} />
                            <meshStandardMaterial color={hairColor} roughness={0.9} />
                        </mesh>
                    </group>
                )}

                {hairStyle === 3 && (
                    <group>
                        {/* Messy / Big Hair */}
                        <mesh position={[0, 0.08, -0.04]} rotation={[-0.3, 0, 0]} castShadow>
                            <sphereGeometry args={[0.135, 32, 32, 0, Math.PI * 2, 0, Math.PI / 1.8]} />
                            <meshStandardMaterial color={hairColor} roughness={0.7} />
                        </mesh>
                        <mesh position={[0.05, 0.05, 0.08]} rotation={[0.1, 0.2, 0]} castShadow>
                            <boxGeometry args={[0.1, 0.1, 0.1]} />
                            <meshStandardMaterial color={hairColor} roughness={0.7} />
                        </mesh>
                        <mesh position={[-0.05, 0.05, 0.08]} rotation={[0.1, -0.2, 0]} castShadow>
                            <boxGeometry args={[0.1, 0.1, 0.1]} />
                            <meshStandardMaterial color={hairColor} roughness={0.7} />
                        </mesh>
                    </group>
                )}
            </group>

            {/* Right Arm - reaching forward onto desk */}
            <group position={[0.18, 0.58, 0]}>
                {/* Upper arm dropping from shoulder toward desk */}
                <mesh position={[0, -0.05, 0.1]} rotation={[-0.6, 0, 0]} castShadow>
                    <capsuleGeometry args={[0.04, 0.22, 4, 8]} />
                    <meshStandardMaterial color={shirtColor} />
                </mesh>
                {/* Forearm resting horizontal on desk surface */}
                <mesh position={[0, -0.1, 0.3]} rotation={[-1.5, 0, 0]} castShadow>
                    <capsuleGeometry args={[0.035, 0.18, 4, 8]} />
                    <meshStandardMaterial color={shirtColor} />
                </mesh>
                {/* Hand resting flat on desk */}
                <group position={[0, -0.1, 0.48]}>
                    {/* Palm - flat oval shape */}
                    <mesh scale={[1, 0.4, 1.4]} castShadow>
                        <sphereGeometry args={[0.04, 16, 16]} />
                        <meshStandardMaterial color="#f3e5ab" />
                    </mesh>
                    {/* Fingers - extending forward on desk */}
                    <mesh position={[0, 0, 0.05]} rotation={[-1.5, 0, 0]} castShadow>
                        <capsuleGeometry args={[0.012, 0.04, 4, 4]} />
                        <meshStandardMaterial color="#f3e5ab" />
                    </mesh>
                    <mesh position={[0.025, 0, 0.045]} rotation={[-1.5, 0, -0.15]} castShadow>
                        <capsuleGeometry args={[0.01, 0.035, 4, 4]} />
                        <meshStandardMaterial color="#f3e5ab" />
                    </mesh>
                    <mesh position={[-0.025, 0, 0.045]} rotation={[-1.5, 0, 0.15]} castShadow>
                        <capsuleGeometry args={[0.01, 0.035, 4, 4]} />
                        <meshStandardMaterial color="#f3e5ab" />
                    </mesh>
                </group>
            </group>

            {/* Left Arm - reaching forward onto desk */}
            <group position={[-0.18, 0.58, 0]}>
                {/* Upper arm dropping from shoulder toward desk */}
                <mesh position={[0, -0.05, 0.1]} rotation={[-0.6, 0, 0]} castShadow>
                    <capsuleGeometry args={[0.04, 0.22, 4, 8]} />
                    <meshStandardMaterial color={shirtColor} />
                </mesh>
                {/* Forearm resting horizontal on desk surface */}
                <mesh position={[0, -0.1, 0.3]} rotation={[-1.5, 0, 0]} castShadow>
                    <capsuleGeometry args={[0.035, 0.18, 4, 8]} />
                    <meshStandardMaterial color={shirtColor} />
                </mesh>
                {/* Hand resting flat on desk */}
                <group position={[0, -0.1, 0.48]}>
                    {/* Palm - flat oval shape */}
                    <mesh scale={[1, 0.4, 1.4]} castShadow>
                        <sphereGeometry args={[0.04, 16, 16]} />
                        <meshStandardMaterial color="#f3e5ab" />
                    </mesh>
                    {/* Fingers - extending forward on desk */}
                    <mesh position={[0, 0, 0.05]} rotation={[-1.5, 0, 0]} castShadow>
                        <capsuleGeometry args={[0.012, 0.04, 4, 4]} />
                        <meshStandardMaterial color="#f3e5ab" />
                    </mesh>
                    <mesh position={[0.025, 0, 0.045]} rotation={[-1.5, 0, -0.15]} castShadow>
                        <capsuleGeometry args={[0.01, 0.035, 4, 4]} />
                        <meshStandardMaterial color="#f3e5ab" />
                    </mesh>
                    <mesh position={[-0.025, 0, 0.045]} rotation={[-1.5, 0, 0.15]} castShadow>
                        <capsuleGeometry args={[0.01, 0.035, 4, 4]} />
                        <meshStandardMaterial color="#f3e5ab" />
                    </mesh>
                </group>
            </group>

            {/* Legs - Composed Thighs (horizontal on chair) and Calves (vertical to floor) */}
            {/* Thighs */}
            <mesh position={[0.1, 0.25, 0.15]} rotation={[Math.PI / 2, 0, 0]}>
                <capsuleGeometry args={[0.05, 0.35, 4, 8]} />
                <meshStandardMaterial color="#2c3e50" />
            </mesh>
            <mesh position={[-0.1, 0.25, 0.15]} rotation={[Math.PI / 2, 0, 0]}>
                <capsuleGeometry args={[0.05, 0.35, 4, 8]} />
                <meshStandardMaterial color="#2c3e50" />
            </mesh>

            {/* Calves */}
            <mesh position={[0.1, 0.05, 0.3]} rotation={[0, 0, 0]}>
                <capsuleGeometry args={[0.04, 0.4, 4, 8]} />
                <meshStandardMaterial color="#2c3e50" />
            </mesh>
            <mesh position={[-0.1, 0.05, 0.3]} rotation={[0, 0, 0]}>
                <capsuleGeometry args={[0.04, 0.4, 4, 8]} />
                <meshStandardMaterial color="#2c3e50" />
            </mesh>
        </group>
    );
};

export default StudentAvatar;

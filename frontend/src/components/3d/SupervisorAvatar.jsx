import React, { useRef } from 'react';
import { useFrame } from '@react-three/fiber';

/**
 * SupervisorAvatar Component
 * A character that walks back and forth in the exam hall.
 */
const SupervisorAvatar = ({ position = [0, 0, 0], pathLength = 8, speed = 0.5, currentAlert }) => {
    const groupRef = useRef();
    const isAlerting = currentAlert && (currentAlert.severity === 'critical' || currentAlert.severity === 'warning');

    useFrame((state) => {
        if (!groupRef.current) return;

        if (isAlerting) {
            // Stop and face the student (at [0, 0, 0])
            groupRef.current.lookAt(0, groupRef.current.position.y, 0);
            // Subtle "angry" bobbing
            groupRef.current.position.y = Math.abs(Math.sin(state.clock.getElapsedTime() * 10)) * 0.02;
            return;
        }

        // Simple back and forth movement along X axis
        const t = state.clock.getElapsedTime() * speed;
        const xOffset = Math.sin(t) * pathLength;

        groupRef.current.position.x = position[0] + xOffset;

        // Face the direction of movement
        const direction = Math.cos(t);
        if (Math.abs(direction) > 0.1) {
            groupRef.current.rotation.y = direction > 0 ? Math.PI / 2 : -Math.PI / 2;
        }

        // Subtle walking "bob"
        groupRef.current.position.y = Math.abs(Math.sin(t * 4)) * 0.05;
    });

    const getMaterialColor = () => {
        if (!isAlerting) return "#2c3e50";
        return currentAlert.severity === 'critical' ? "#ff0000" : "#ff9f43";
    };

    return (
        <group ref={groupRef} position={position}>
            {/* Organic Torso */}
            <mesh position={[0, 1.1, 0]} scale={[1, 1, 0.6]} castShadow>
                <capsuleGeometry args={[0.22, 0.45, 16, 16]} />
                <meshStandardMaterial color={getMaterialColor()} />
            </mesh>

            {/* Neck Joint */}
            <mesh position={[0, 1.45, 0]} castShadow>
                <sphereGeometry args={[0.08, 16, 16]} />
                <meshStandardMaterial color={isAlerting ? "#ff7675" : "#f3e5ab"} />
            </mesh>


            {/* Head */}
            <mesh position={[0, 1.6, 0]} castShadow>
                <sphereGeometry args={[0.15, 32, 32]} />
                <meshStandardMaterial color={isAlerting ? "#ff7675" : "#f3e5ab"} />
            </mesh>

            {/* Organic Shoulder Joints */}
            <mesh position={[0.26, 1.25, 0]} castShadow>
                <sphereGeometry args={[0.07, 16, 16]} />
                <meshStandardMaterial color={getMaterialColor()} />
            </mesh>
            <mesh position={[-0.26, 1.25, 0]} castShadow>
                <sphereGeometry args={[0.07, 16, 16]} />
                <meshStandardMaterial color={getMaterialColor()} />
            </mesh>

            {/* Arms - Organic Capsules */}
            <mesh position={[0.26, 0.95, 0]} castShadow>
                <capsuleGeometry args={[0.05, 0.4, 4, 16]} />
                <meshStandardMaterial color={getMaterialColor()} />
            </mesh>
            <mesh position={[-0.26, 0.95, 0]} castShadow>
                <capsuleGeometry args={[0.05, 0.4, 4, 16]} />
                <meshStandardMaterial color={getMaterialColor()} />
            </mesh>

            {/* Hands - Organic Hand Shapes */}
            <mesh position={[0.26, 0.72, 0]} castShadow>
                <sphereGeometry args={[0.05, 16, 16]} scale={[1, 1.2, 0.6]} />
                <meshStandardMaterial color={isAlerting ? "#ff7675" : "#f3e5ab"} />
            </mesh>
            <mesh position={[-0.26, 0.72, 0]} castShadow>
                <sphereGeometry args={[0.05, 16, 16]} scale={[1, 1.2, 0.6]} />
                <meshStandardMaterial color={isAlerting ? "#ff7675" : "#f3e5ab"} />
            </mesh>

            {/* Hip Joints */}
            <mesh position={[0.13, 0.75, 0]} castShadow>
                <sphereGeometry args={[0.08, 16, 16]} />
                <meshStandardMaterial color="#1a252f" />
            </mesh>
            <mesh position={[-0.13, 0.75, 0]} castShadow>
                <sphereGeometry args={[0.08, 16, 16]} />
                <meshStandardMaterial color="#1a252f" />
            </mesh>

            {/* Legs - Organic Capsules */}
            <mesh position={[0.13, 0.4, 0]} castShadow>
                <capsuleGeometry args={[0.065, 0.6, 4, 16]} />
                <meshStandardMaterial color="#1a252f" />
            </mesh>
            <mesh position={[-0.13, 0.4, 0]} castShadow>
                <capsuleGeometry args={[0.065, 0.6, 4, 16]} />
                <meshStandardMaterial color="#1a252f" />
            </mesh>
        </group>
    );
};

export default SupervisorAvatar;

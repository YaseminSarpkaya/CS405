function multiplyMatrices(matrixA, matrixB) {
    var result = [];

    for (var i = 0; i < 4; i++) {
        result[i] = [];
        for (var j = 0; j < 4; j++) {
            var sum = 0;
            for (var k = 0; k < 4; k++) {
                sum += matrixA[i * 4 + k] * matrixB[k * 4 + j];
            }
            result[i][j] = sum;
        }
    }

    // Flatten the result array
    return result.reduce((a, b) => a.concat(b), []);
}
function createIdentityMatrix() {
    return new Float32Array([
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    ]);
}
function createScaleMatrix(scale_x, scale_y, scale_z) {
    return new Float32Array([
        scale_x, 0, 0, 0,
        0, scale_y, 0, 0,
        0, 0, scale_z, 0,
        0, 0, 0, 1
    ]);
}

function createTranslationMatrix(x_amount, y_amount, z_amount) {
    return new Float32Array([
        1, 0, 0, x_amount,
        0, 1, 0, y_amount,
        0, 0, 1, z_amount,
        0, 0, 0, 1
    ]);
}

function createRotationMatrix_Z(radian) {
    return new Float32Array([
        Math.cos(radian), -Math.sin(radian), 0, 0,
        Math.sin(radian), Math.cos(radian), 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    ])
}

function createRotationMatrix_X(radian) {
    return new Float32Array([
        1, 0, 0, 0,
        0, Math.cos(radian), -Math.sin(radian), 0,
        0, Math.sin(radian), Math.cos(radian), 0,
        0, 0, 0, 1
    ])
}

function createRotationMatrix_Y(radian) {
    return new Float32Array([
        Math.cos(radian), 0, Math.sin(radian), 0,
        0, 1, 0, 0,
        -Math.sin(radian), 0, Math.cos(radian), 0,
        0, 0, 0, 1
    ])
}

function getTransposeMatrix(matrix) {
    return new Float32Array([
        matrix[0], matrix[4], matrix[8], matrix[12],
        matrix[1], matrix[5], matrix[9], matrix[13],
        matrix[2], matrix[6], matrix[10], matrix[14],
        matrix[3], matrix[7], matrix[11], matrix[15]
    ]);
}

const vertexShaderSource = `
attribute vec3 position;
attribute vec3 normal; // Normal vector for lighting

uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform mat4 normalMatrix;

uniform vec3 lightDirection;

varying vec3 vNormal;
varying vec3 vLightDirection;

void main() {
    vNormal = vec3(normalMatrix * vec4(normal, 0.0));
    vLightDirection = lightDirection;

    gl_Position = vec4(position, 1.0) * projectionMatrix * modelViewMatrix; 
}

`

const fragmentShaderSource = `
precision mediump float;

uniform vec3 ambientColor;
uniform vec3 diffuseColor;
uniform vec3 specularColor;
uniform float shininess;

varying vec3 vNormal;
varying vec3 vLightDirection;

void main() {
    vec3 normal = normalize(vNormal);
    vec3 lightDir = normalize(vLightDirection);
    
    // Ambient component
    vec3 ambient = ambientColor;

    // Diffuse component
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * diffuseColor;

    // Specular component (view-dependent)
    vec3 viewDir = vec3(0.0, 0.0, 1.0); // Assuming the view direction is along the z-axis
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
    vec3 specular = spec * specularColor;

    gl_FragColor = vec4(ambient + diffuse + specular, 1.0);
}

`

/**
 * @WARNING DO NOT CHANGE ANYTHING ABOVE THIS LINE
 */



/**
 * 
 * @TASK1 Calculate the model view matrix by using the chatGPT
 */

function getChatGPTModelViewMatrix() {
    const transformationMatrix = new Float32Array([
        // you should paste the response of the chatGPT here:
        
        0.433, -0.25, 0.25, 0.3,
        0.2165, 0.375, -0.433, -0.25,
        -0.125, 0.2165, 0.75, 0,
        0, 0, 0, 1

    ]);
    return getTransposeMatrix(transformationMatrix);
}


/**
 * 
 * @TASK2 Calculate the model view matrix by using the given 
 * transformation methods and required transformation parameters
 * stated in transformation-prompt.txt
 */
function getModelViewMatrix() {
    // Translation matrix
    const translationMatrix = new Float32Array([
        1, 0, 0, 0.3,
        0, 1, 0, -0.25,
        0, 0, 1, 0,
        0, 0, 0, 1
    ]);

    // Scaling matrix
    const scalingMatrix = new Float32Array([
        0.5, 0, 0, 0,
        0, 0.5, 0, 0,
        0, 0, 0.5, 0,
        0, 0, 0, 1
    ]);

    // Conversion of rotation angles to radians
    const toRadians = angle => (angle * Math.PI) / 180;

    // Rotation matrices
    const rotationX = new Float32Array([
        1, 0, 0, 0,
        0, Math.cos(toRadians(30)), -Math.sin(toRadians(30)), 0,
        0, Math.sin(toRadians(30)), Math.cos(toRadians(30)), 0,
        0, 0, 0, 1
    ]);

    const rotationY = new Float32Array([
        Math.cos(toRadians(45)), 0, Math.sin(toRadians(45)), 0,
        0, 1, 0, 0,
        -Math.sin(toRadians(45)), 0, Math.cos(toRadians(45)), 0,
        0, 0, 0, 1
    ]);

    const rotationZ = new Float32Array([
        Math.cos(toRadians(60)), -Math.sin(toRadians(60)), 0, 0,
        Math.sin(toRadians(60)), Math.cos(toRadians(60)), 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    ]);

    // Multiplying matrices: Translation * RotationZ * RotationY * RotationX * Scaling
    const intermediateMatrix = new Float32Array(16);
    multiplyMatrices(translationMatrix, rotationZ, intermediateMatrix);
    multiplyMatrices(intermediateMatrix, rotationY, intermediateMatrix);
    multiplyMatrices(intermediateMatrix, rotationX, intermediateMatrix);
    multiplyMatrices(intermediateMatrix, scalingMatrix, intermediateMatrix);

    return intermediateMatrix;
}

// Function to multiply two 4x4 matrices
function multiplyMatrices(a, b, result) {
    for (let i = 0; i < 4; i++) {
        for (let j = 0; j < 4; j++) {
            result[i * 4 + j] =
                a[i * 4] * b[j] +
                a[i * 4 + 1] * b[4 + j] +
                a[i * 4 + 2] * b[8 + j] +
                a[i * 4 + 3] * b[12 + j];
        }
    }
}



/**
 * 
 * @TASK3 Ask CHAT-GPT to animate the transformation calculated in 
 * task2 infinitely with a period of 10 seconds. 
 * First 5 seconds, the cube should transform from its initial 
 * position to the target position.
 * The next 5 seconds, the cube should return to its initial position.
 */function getPeriodicMovement(startTime) {
    // Define the initial and target transformation matrices
    const initialMatrix = getChatGPTModelViewMatrix();
    const targetMatrix = getModelViewMatrix(); // Use the matrix from Task 2

    // Function to interpolate between two matrices based on a progress value
    function interpolateMatrices(matrixA, matrixB, progress) {
        const result = new Float32Array(16);
        for (let i = 0; i < 16; i++) {
            result[i] = matrixA[i] + (matrixB[i] - matrixA[i]) * progress;
        }
        return result;
    }

    // Calculate elapsed time
    const elapsed = (Date.now() - startTime) / 1000;

    // Duration for each phase of the animation (in seconds)
    const duration = 5;

    // Calculate phase and progress within the phase
    const phase = elapsed % (2 * duration);
    const progress = phase < duration ? phase / duration : 1 - ((phase - duration) / duration);

    // Interpolate between initial and target matrices based on the phase and progress
    const modelViewMatrix = interpolateMatrices(initialMatrix, targetMatrix, progress);

    return modelViewMatrix;
}


// Function to interpolate between two 4x4 matrices
function interpolateMatrices(matrix1, matrix2, progress) {
    const interpolatedMatrix = new Float32Array(16);

    for (let i = 0; i < 16; i++) {
        interpolatedMatrix[i] = matrix1[i] + (matrix2[i] - matrix1[i]) * progress;
    }

    return interpolatedMatrix;
}


// Adapted from collidingScopes/video-to-pixel-art; MIT, Alan Ang 2024.
precision mediump float;
    varying vec2 vTexCoord;
    uniform sampler2D uTexture;
    uniform vec2 resolution;
    uniform float pixelSize;
    uniform float ditherFactor;
    uniform int paletteChoice;
    uniform float edgeThreshold;
    uniform float edgeIntensity;
    uniform vec3 edgeColor;

    uniform vec3 palette[16];
    uniform int paletteCount;

    // Sobel operator kernels
    mat3 sobelX = mat3(
        -1.0, 0.0, 1.0,
        -2.0, 0.0, 2.0,
        -1.0, 0.0, 1.0
    );

    mat3 sobelY = mat3(
        -1.0, -2.0, -1.0,
         0.0,  0.0,  0.0,
         1.0,  2.0,  1.0
    );

    // Helper function to get grayscale value
    float getLuminance(vec3 color) {
        return dot(color, vec3(0.299, 0.587, 0.114));
    }

    // Edge detection function
    float detectEdge(vec2 coord) {
        float pixelWidth = 1.0 / resolution.x;
        float pixelHeight = 1.0 / resolution.y;
        
        float gx = 0.0;
        float gy = 0.0;
        
        // Apply Sobel operator
        for(int i = -1; i <= 1; i++) {
            for(int j = -1; j <= 1; j++) {
                vec2 offset = vec2(float(i) * pixelWidth, float(j) * pixelHeight);
                vec3 color = texture2D(uTexture, coord + offset).rgb;
                float luminance = getLuminance(color);
                
                gx += luminance * sobelX[i+1][j+1];
                gy += luminance * sobelY[i+1][j+1];
            }
        }
        
        return sqrt(gx * gx + gy * gy);
    }

    vec3 findClosestColor(vec3 color) {
        float best = 1000.0;
        vec3 chosen = palette[0];
        for (int i = 0; i < 16; i++) {
            if (i >= paletteCount) break;
            float d = distance(color, palette[i]);
            if (d < best) { best = d; chosen = palette[i]; }
        }
        return chosen;
    }

    float mod2(float x, float y) {
        return x - y * floor(x/y);
    }

    // 4x4 Bayer matrix indexed using mod2
    float getBayerValue(vec2 coord) {
        float x = mod2(coord.x, 4.0);
        float y = mod2(coord.y, 4.0);
        
        if(x < 1.0) {
            if(y < 1.0) return 0.0/16.0;
            else if(y < 2.0) return 12.0/16.0;
            else if(y < 3.0) return 3.0/16.0;
            else return 15.0/16.0;
        } 
        else if(x < 2.0) {
            if(y < 1.0) return 8.0/16.0;
            else if(y < 2.0) return 4.0/16.0;
            else if(y < 3.0) return 11.0/16.0;
            else return 7.0/16.0;
        }
        else if(x < 3.0) {
            if(y < 1.0) return 2.0/16.0;
            else if(y < 2.0) return 14.0/16.0;
            else if(y < 3.0) return 1.0/16.0;
            else return 13.0/16.0;
        }
        else {
            if(y < 1.0) return 10.0/16.0;
            else if(y < 2.0) return 6.0/16.0;
            else if(y < 3.0) return 9.0/16.0;
            else return 5.0/16.0;
        }
    }

    void main() {
        vec2 pixelatedCoord = (floor(vTexCoord * resolution / pixelSize) + 0.5) * pixelSize / resolution;
        vec4 color = texture2D(uTexture, pixelatedCoord);

        // Edge detection
        float edge = detectEdge(pixelatedCoord);
        bool isEdge = edge > edgeThreshold;

        // Get the dither threshold using screen coordinates
        float threshold = getBayerValue(floor(vTexCoord * resolution / pixelSize));

        // Apply dithering by adjusting the color before quantization
        vec3 adjustedColor = color.rgb + (threshold - 0.5) * ditherFactor;
        
        // Clamp the adjusted color
        adjustedColor = clamp(adjustedColor, 0.0, 1.0);
        
        // Find the closest color in the palette for the adjusted color
        vec3 quantizedColor = findClosestColor(adjustedColor);

        // Apply edge highlighting
        if (isEdge) {
            quantizedColor = mix(quantizedColor, edgeColor, edgeIntensity);
        }

        gl_FragColor = vec4(quantizedColor, 1.0);
    }

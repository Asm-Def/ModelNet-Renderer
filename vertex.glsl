#version 430 core

layout (location=0) in vec3 position;
layout (location=1) in vec3 normal;
out vec4 vertexColor;
out vec3 Normal;
out vec4 Position;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    Position = view * model * vec4(position.xyz, 1.0);
    //gl_Position = projection * view * Position;
    gl_Position = projection * Position;
    Normal = mat3(transpose(inverse(view * model))) * normal;
    //vertexColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);
    vertexColor = vec4(vec3(0.8f), 1.0f);
}
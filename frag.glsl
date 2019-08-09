#version 430 core
in vec4 vertexColor;
in vec3 Normal;
in vec4 Position;
out vec4 color;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    vec3 light_pos = vec3(0.0f, 2.0f, -2.0f);
    vec3 light_color = vec3(1.0f, 1.0f, 1.0f);
    float ambientStrength = 0.3f;

    vec3 ambient = ambientStrength * light_color;
    vec3 light_dir = normalize(light_pos - Position.xyz);
    //float tmp = dot(normalize(Normal), normalize(Normal));
    float tmp = Normal.z;
    float light_rate;
    vec3 normal = normalize(Normal);
    if(tmp > 0) normal = -normal;
    float diff = max(dot(normal, light_dir), 0.0) * (1 - ambientStrength);
    vec3 diffuse = diff * light_color;
    light_rate = dot(normal, normalize(light_pos.xyz - Position.xyz));
    color = vec4((ambient + diffuse) * vertexColor.rgb, 0.3f);
    //color = vec4(normal, 1.0f);
}
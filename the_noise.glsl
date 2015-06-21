#version 120

// concrete noise based on https://www.shadertoy.com/view/4lfGRs by S.Guillitte
// Simplex3D noise based on https://www.shadertoy.com/view/XtBGDG by Lallis
// FBM noise by iq
// Fractal Noise based on https://www.shadertoy.com/view/Msf3Wr by mu6k
// Value Noise based on https://www.shadertoy.com/view/lsf3WH by iq
// Gradient Noise based on https://www.shadertoy.com/view/XdXGW8 by iq
// Worley Noise  based on http://glslsandbox.com/e#25658.1
// Ridged Noise based on https://www.shadertoy.com/view/ldj3Dw by nimitz
// Perlin Noise based on https://www.shadertoy.com/view/MllGzs by guil
// Perlin v2 Noise based on https://www.shadertoy.com/view/MlS3z1 byRenoM
// Crawling Noise based on https://www.shadertoy.com/view/lslXRS by nimitz

// License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.

#define PI 3.14159265359
#define tau 6.2831853
#define r_iter 6.

uniform float adsk_result_w, adsk_result_h, adsk_time, adsk_result_frameratio;
vec2 resolution = vec2(adsk_result_w, adsk_result_h);

// global uniforms
uniform float speed;
uniform float offset;
uniform float scale;
uniform float aspect;
uniform float rot;
uniform vec2 pos;
uniform int noise_type;


float time = adsk_time *.05 * speed + offset;


// concrete uniforms
uniform float c_detail;
uniform int c_noise_itt;

// fractal noise uniforms
uniform float f_detail;

// value noise uniforms
uniform int v_noise_type;

// gradient noise v1 uniforms
uniform float g1_detail;

// gradient noise v2 uniforms
uniform float g2_detail;

// worley uniform
uniform float w_edge_detail;

// ridged noise uniforms
uniform int r_noise_type;
uniform float r_detail;

// perlin v1 uniforms
uniform int p1_itt;
uniform float perlinv1_v;

// Plasma uniforms
uniform int plasma_iter;
uniform float plasma_detail;

// Marble uniforms
uniform int marble_iter;
uniform float marble_detail;

// Wood uniforms
uniform int wood_iter;
uniform float wood_detail;

// Clouds uniforms
uniform int cloud_iter;
uniform float cloud_detail;

// Crawling uniforms
uniform int crawling_iter;
uniform float crawling_detail;
uniform float crawling_displace;

// start concrete noise
float hash ( in vec2 p ) 
{
    return fract(sin(p.x*15.32+p.y*35.78) * 43758.23);
}

vec2 hash2 ( vec2 p )
{
	return vec2(hash(p*.754),hash(1.5743*p.yx+4.5891))-.5;
}

vec2 noise(vec2 x)
{
	vec2 add = vec2(1.0, .0);
	vec2 p = floor(x);
    vec2 f = fract(x);
    f = f*f*(3.0-2.0*f);
    return mix(mix( hash2(p), hash2(p + add.xy),f.x), mix( hash2(p + add.yx), hash2(p + add.xx),f.x),f.y);
}

vec2 fbm(vec2 x)
{
    vec2 r = x;
    float a = 1.;
    for (int i = 0; i < c_noise_itt; i++)
    {
        r += noise(r*a)/a;
        a = c_detail;
    }     
    return (r-x)*sqrt(a);
}
// end concrete noise



// start Simplex3D 
float noise3D(vec3 p)
{
	return fract(sin(dot(p ,vec3(12.9898,78.233,128.852))) * 43758.5453)*2.0-1.0;
}

float simplex3D(vec3 p)
{
	
	float f3 = 1.0/3.0;
	float s = (p.x+p.y+p.z)*f3;
	int i = int(floor(p.x+s));
	int j = int(floor(p.y+s));
	int k = int(floor(p.z+s));
	
	float g3 = 1.0/6.0;
	float t = float((i+j+k))*g3;
	float x0 = float(i)-t;
	float y0 = float(j)-t;
	float z0 = float(k)-t;
	x0 = p.x-x0;
	y0 = p.y-y0;
	z0 = p.z-z0;
	
	int i1,j1,k1;
	int i2,j2,k2;
	
	if(x0>=y0)
	{
		if(y0>=z0){ i1=1; j1=0; k1=0; i2=1; j2=1; k2=0; } // X Y Z order
		else if(x0>=z0){ i1=1; j1=0; k1=0; i2=1; j2=0; k2=1; } // X Z Y order
		else { i1=0; j1=0; k1=1; i2=1; j2=0; k2=1; }  // Z X Z order
	}
	else 
	{ 
		if(y0<z0) { i1=0; j1=0; k1=1; i2=0; j2=1; k2=1; } // Z Y X order
		else if(x0<z0) { i1=0; j1=1; k1=0; i2=0; j2=1; k2=1; } // Y Z X order
		else { i1=0; j1=1; k1=0; i2=1; j2=1; k2=0; } // Y X Z order
	}
	
	float x1 = x0 - float(i1) + g3; 
	float y1 = y0 - float(j1) + g3;
	float z1 = z0 - float(k1) + g3;
	float x2 = x0 - float(i2) + 2.0*g3; 
	float y2 = y0 - float(j2) + 2.0*g3;
	float z2 = z0 - float(k2) + 2.0*g3;
	float x3 = x0 - 1.0 + 3.0*g3; 
	float y3 = y0 - 1.0 + 3.0*g3;
	float z3 = z0 - 1.0 + 3.0*g3;	
				 
	vec3 ijk0 = vec3(i,j,k);
	vec3 ijk1 = vec3(i+i1,j+j1,k+k1);	
	vec3 ijk2 = vec3(i+i2,j+j2,k+k2);
	vec3 ijk3 = vec3(i+1,j+1,k+1);	
            
	vec3 gr0 = normalize(vec3(noise3D(ijk0),noise3D(ijk0*2.01),noise3D(ijk0*2.02)));
	vec3 gr1 = normalize(vec3(noise3D(ijk1),noise3D(ijk1*2.01),noise3D(ijk1*2.02)));
	vec3 gr2 = normalize(vec3(noise3D(ijk2),noise3D(ijk2*2.01),noise3D(ijk2*2.02)));
	vec3 gr3 = normalize(vec3(noise3D(ijk3),noise3D(ijk3*2.01),noise3D(ijk3*2.02)));
	
	float n0 = 0.0;
	float n1 = 0.0;
	float n2 = 0.0;
	float n3 = 0.0;

	float t0 = 0.5 - x0*x0 - y0*y0 - z0*z0;
	if(t0>=0.0)
	{
		t0*=t0;
		n0 = t0 * t0 * dot(gr0, vec3(x0, y0, z0));
	}
	float t1 = 0.5 - x1*x1 - y1*y1 - z1*z1;
	if(t1>=0.0)
	{
		t1*=t1;
		n1 = t1 * t1 * dot(gr1, vec3(x1, y1, z1));
	}
	float t2 = 0.5 - x2*x2 - y2*y2 - z2*z2;
	if(t2>=0.0)
	{
		t2 *= t2;
		n2 = t2 * t2 * dot(gr2, vec3(x2, y2, z2));
	}
	float t3 = 0.5 - x3*x3 - y3*y3 - z3*z3;
	if(t3>=0.0)
	{
		t3 *= t3;
		n3 = t3 * t3 * dot(gr3, vec3(x3, y3, z3));
	}
	return 96.0*(n0+n1+n2+n3);
	
}
// end Simplex3D

// stat FBM
float fbm(vec3 p)
{
	float f;
    f  = 0.50000*simplex3D( p ); p = p*2.01;
    f += 0.25000*simplex3D( p ); p = p*2.02; //from iq
    f += 0.12500*simplex3D( p ); p = p*2.03;
    f += 0.06250*simplex3D( p ); p = p*2.04;
    f += 0.03125*simplex3D( p );
	return f;
}
// end FBM

float hash(float x)
{
	return fract(sin(cos(x*12.13)*19.123)*17.321);
}

// start Fractal Noise
float fn_noise(vec2 p)
{
	vec2 pm = mod(p,1.0);
	vec2 pd = p-pm;
	float v0=hash(pd.x+pd.y*41.0);
	float v1=hash(pd.x+1.0+pd.y*41.0);
	float v2=hash(pd.x+pd.y*41.0+41.0);
	float v3=hash(pd.x+pd.y*41.0+42.0);
	v0 = mix(v0,v1,smoothstep(0.0,1.0,pm.x));
	v2 = mix(v2,v3,smoothstep(0.0,1.0,pm.x));
	return mix(v0,v2,smoothstep(0.0,1.0,pm.y));
}
// end Fractal Noise

// start Value Noise
float v_hash( vec2 p )
{
	float h = dot(p,vec2(127.1,311.7));
	
    return -1.0 + 2.0*fract(sin(h)*43758.5453123 + time);
}

float v_noise( in vec2 p )
{
    vec2 i = floor( p );
    vec2 f = fract( p );
	vec2 u = f*f*(3.0-2.0*f);
    return mix( mix( v_hash( i + vec2(0.0,0.0) ), 
                     v_hash( i + vec2(1.0,0.0) ), u.x),
                mix( v_hash( i + vec2(0.0,1.0) ), 
                     v_hash( i + vec2(1.0,1.0) ), u.x), u.y);
}
// end Value Noise

// start Perlin Noise
float vnoise(vec2 x)//Value noise
{
    vec2 p = floor(x);
    vec2 f = fract(x);
    f = f*f*(3.0-2.0*f);
    
    return  2.*mix(mix( hash(p),hash(p + vec2(1.,0.)),f.x),
                  mix( hash(p+vec2(0.,1.)), hash(p+1.),f.x),f.y)-1.;
            
}
mat2 m2= mat2(.8,.6,-.6,.8);

float dvnoise(vec2 p)//Value noise + value noise with rotation
{
    return .5*(vnoise(p - time)+vnoise(m2*p + time));    
}

float noise5( vec2 p)
{
    return dvnoise(p);
}

float fbm5( vec2 p ) {
	
	float f=5.0, a= perlinv1_v;
   
	float r = 0.0;	
    for(int i = 0;i<p1_itt;i++)
	{	
		r += a	* abs(noise5( p*f ) );       
		a *= .5; f *= 2.0;
	}
	return r/2.;
}
// end Perlin Noise

// start Gradient Noise
vec2 g_hash( vec2 p )
{
	p = vec2( dot(p,vec2(127.1,311.7)),
			  dot(p,vec2(269.5,183.3)) );
	return -1.0 + 2.0*fract(sin(p)*43758.5453123);
}

float g_noise( in vec2 p )
{
    vec2 i = floor( p );
    vec2 f = fract( p );
	vec2 u = f*f*(3.0-2.0*f);
    return mix( mix( dot( g_hash( i + vec2(0.0,0.0) ), f - vec2(0.0,0.0) ), 
                     dot( g_hash( i + vec2(1.0,0.0) ), f - vec2(1.0,0.0) ), u.x),
                mix( dot( g_hash( i + vec2(0.0,1.0) ), f - vec2(0.0,1.0) ), 
                     dot( g_hash( i + vec2(1.0,1.0) ), f - vec2(1.0,1.0) ), u.x), u.y);
}

// start Worley Noise
float w_length2(vec2 p)
{
	return dot(p, p);
}

float w_noise(vec2 p){
	return fract(sin(fract(sin(p.x) * (43.13311)) + p.y) * 31.0011);
}

float w_worley(vec2 p) {
	float d = 1e30;
	for (int xo = -1; xo <= 1; ++xo) {
		for (int yo = -1; yo <= 1; ++yo) {
			vec2 tp = floor(p) + vec2(xo, yo);
			d = min(d, w_length2(p - tp - vec2(w_noise(tp))));
		}
	}
	return exp(-5.0*abs(1.0 * w_edge_detail * d - 1.0));
}

float w_fworley(vec2 p) {
	return sqrt(sqrt(sqrt(w_worley(p) * sqrt(w_worley(p * scale)) *	sqrt(sqrt(w_worley(p))))));
}
// end Worley noise

// start Ridged Noise
mat2 makem2(float theta)
{
	float c = cos(theta);
	float s = sin(theta);
	return mat2(c,-s,s,c);
}

// start Perlin v2 Noise
float p2_rand(vec2 uv)
{
    float dt = dot(uv, vec2(12.9898, 78.233));
	return fract(sin(mod(dt, PI / 2.0)) * 43758.5453);
}

float plasma_turbulence(vec2 uv, float octave, int id)
{
    float col = 0.0;
    vec2 xy;
    vec2 frac;
    vec2 tmp1;
    vec2 tmp2;
    float i2;
    float amp;
    float maxOct = octave;
    for (int i = 0; i < plasma_iter; i++)
    {
        amp = maxOct / octave;
        i2 = float(i);
        xy = id == 1 || id == 4? (uv + 50.0 * float(id) * time / (1.0 + i2)) / octave : uv / octave;
        frac = fract(xy);
        tmp1 = mod(floor(xy) + resolution.xy, resolution.xy);
        tmp2 = mod(tmp1 + resolution.xy - 1.0, resolution.xy);
        col += frac.x * frac.y * p2_rand(tmp1) / amp;
        col += frac.x * (1.0 - frac.y) * p2_rand(vec2(tmp1.x, tmp2.y)) / amp;
        col += (1.0 - frac.x) * frac.y * p2_rand(vec2(tmp2.x, tmp1.y)) / amp;
        col += (1.0 - frac.x) * (1.0 - frac.y) * p2_rand(tmp2) / amp;
        octave /= 2.0;
    }
    return (col);
}

float marble_turbulence(vec2 uv, float octave, int id)
{
    float col = 0.0;
    vec2 xy;
    vec2 frac;
    vec2 tmp1;
    vec2 tmp2;
    float i2;
    float amp;
    float maxOct = octave;
    for (int i = 0; i < marble_iter; i++)
    {
        amp = maxOct / octave;
        i2 = float(i);
        xy = id == 1 || id == 4? (uv + 50.0 * float(id) * time / (1.0 + i2)) / octave : uv / octave;
        frac = fract(xy);
        tmp1 = mod(floor(xy) + resolution.xy, resolution.xy);
        tmp2 = mod(tmp1 + resolution.xy - 1.0, resolution.xy);
        col += frac.x * frac.y * p2_rand(tmp1) / amp;
        col += frac.x * (1.0 - frac.y) * p2_rand(vec2(tmp1.x, tmp2.y)) / amp;
        col += (1.0 - frac.x) * frac.y * p2_rand(vec2(tmp2.x, tmp1.y)) / amp;
        col += (1.0 - frac.x) * (1.0 - frac.y) * p2_rand(tmp2) / amp;
        octave /= 2.0;
    }
    return (col);
}

float cloud_turbulence(vec2 uv, float octave, int id)
{
    float col = 0.0;
    vec2 xy;
    vec2 frac;
    vec2 tmp1;
    vec2 tmp2;
    float i2;
    float amp;
    float maxOct = octave;
    for (int i = 0; i < cloud_iter; i++)
    {
        amp = maxOct / octave;
        i2 = float(i);
        xy = id == 1 || id == 4? (uv + 50.0 * float(id) * time / (1.0 + i2)) / octave : uv / octave;
        frac = fract(xy);
        tmp1 = mod(floor(xy) + resolution.xy, resolution.xy);
        tmp2 = mod(tmp1 + resolution.xy - 1.0, resolution.xy);
        col += frac.x * frac.y * p2_rand(tmp1) / amp;
        col += frac.x * (1.0 - frac.y) * p2_rand(vec2(tmp1.x, tmp2.y)) / amp;
        col += (1.0 - frac.x) * frac.y * p2_rand(vec2(tmp2.x, tmp1.y)) / amp;
        col += (1.0 - frac.x) * (1.0 - frac.y) * p2_rand(tmp2) / amp;
        octave /= 2.0;
    }
    return (col);
}

float wood_turbulence(vec2 uv, float octave, int id)
{
    float col = 0.0;
    vec2 xy;
    vec2 frac;
    vec2 tmp1;
    vec2 tmp2;
    float i2;
    float amp;
    float maxOct = octave;
    for (int i = 0; i < wood_iter; i++)
    {
        amp = maxOct / octave;
        i2 = float(i);
        xy = id == 1 || id == 4? (uv + 50.0 * float(id) * time / (1.0 + i2)) / octave : uv / octave;
        frac = fract(xy);
        tmp1 = mod(floor(xy) + resolution.xy, resolution.xy);
        tmp2 = mod(tmp1 + resolution.xy - 1.0, resolution.xy);
        col += frac.x * frac.y * p2_rand(tmp1) / amp;
        col += frac.x * (1.0 - frac.y) * p2_rand(vec2(tmp1.x, tmp2.y)) / amp;
        col += (1.0 - frac.x) * frac.y * p2_rand(vec2(tmp2.x, tmp1.y)) / amp;
        col += (1.0 - frac.x) * (1.0 - frac.y) * p2_rand(tmp2) / amp;
        octave /= 2.0;
    }
    return (col);
}

vec3 p2_clouds(vec2 uv)
{
    float col = cloud_turbulence(uv, 128.0 * cloud_detail, 1) * 0.75;
    return (vec3(col - 0.1));
}

vec3 p2_marble(vec2 uv)
{
	vec2 period = vec2(3.0, 4.0);
    vec2 turb = vec2(4.0, 64.0 * marble_detail);
    float xy = uv.x * period.x / resolution.y + uv.y * period.y / resolution.x + turb.x * marble_turbulence(uv, turb.y, 2);
    float col = abs(sin(xy * PI)) * 0.75;
    return (vec3(col));
}

vec3 p2_wood(vec2 uv)
{
    vec2 iR = resolution.xy;
    float period = 3.5 * wood_detail;
    vec2 turb = vec2(0.04, 16.0);
    vec2 xy;
    xy.x = (uv.x - iR.x / 2.0) / iR.y;
    xy.y = (uv.y - iR.y / 2.0) / iR.y;
	xy.x += .88;
	xy.y += 0.5;
    float dist = length(xy) + turb.x * wood_turbulence(uv, turb.y, 3);
    float col = 0.5 * abs(sin(2.0 * period * dist * PI));
    return (vec3(col));
}

vec3 p2_plasma(vec2 uv)
{
	vec2 period = vec2(0.0, 0.0);
    vec2 turb = vec2(1.0, 128.0 * plasma_detail);
    float xy = uv.x * period.x / resolution.y + uv.y * period.y / resolution.x + turb.x * plasma_turbulence(uv, turb.y, 4);
    float col = abs(sin(xy * PI)) * 0.75;
    return (vec3(1. - col));
}

// start Crawling Noise
vec2 crawling_hash( vec2 p )
{
	p = vec2( dot(p,vec2(127.1,110.7)),
			  dot(p,vec2(269.5,45.34)) );
	return 2.0*fract(sin(p)*4378.5453123);
}

float crawling_noise( in vec2 p )
{
    vec2 i = floor( p );
    vec2 f = fract( p );
	vec2 u = f*f*(3.0-2.0*f);
    return mix( mix( dot( crawling_hash( i + vec2(0.0,0.0) ), f - vec2(0.0,0.0) ), 
                     dot( crawling_hash( i + vec2(1.0,0.0) ), f - vec2(1.0,0.0) ), u.x),
                mix( dot( crawling_hash( i + vec2(0.0,1.0) ), f - vec2(0.0,1.0) ), 
                     dot( crawling_hash( i + vec2(1.0,1.0) ), f - vec2(1.0,1.0) ), u.x), u.y);
}

vec2 crawling_gradn(vec2 p)
{
	float ep = 0.5 * crawling_detail;
	float gradx = crawling_noise(vec2(p.x+ep,p.y))-crawling_noise(vec2(p.x-ep,p.y));
	float grady = crawling_noise(vec2(p.x,p.y+ep))-crawling_noise(vec2(p.x,p.y-ep));
	return vec2(gradx,grady);
}

float crawling_flow(in vec2 p)
{
	time *= 0.05;
	float lava_z=2.;
	float rz = 0.;
	vec2 bp = p;
	for (float i= 1.;i < crawling_iter + 1;i++ )
	{
		//secondary flow speed (speed of the perceived flow)
		bp += time*1.9;
		
		//displacement field (try changing time multiplier)
		vec2 gr = crawling_gradn(i*p*.34+time);
		
		//rotation of the displacement field
		gr*=makem2(time*6.-(0.05*p.x+0.03*p.y)*40.);
		
		//displace the system
		p += gr*.5 * crawling_displace;
		
		//add noise octave
		rz+= (sin(crawling_noise(p)*7.)*0.5+0.5)/lava_z;
		
		//blend factor (blending displaced system with base system)
		//you could call this advection factor (.5 being low, .95 being high)
		p = mix(bp,p,.9);
		
		//intensity scaling
		lava_z *= 1.4;
		//octave scaling
		p *= 2.;
		bp *= 1.9;
	}
	return rz;	
}
// end Crawling Noise

void main()
{
	vec2 uv = (gl_FragCoord.xy / resolution.xy) - pos;
    vec4 col = vec4(0.0);
	uv.x *= adsk_result_frameratio;
	float rad_rot = (rot+180.0) * PI / 180.0; 
	mat2 rotation = mat2( cos(-rad_rot), -sin(-rad_rot), sin(-rad_rot), cos(-rad_rot));
	uv *= rotation;
	uv.x *= aspect;
	uv *= scale;

	if ( noise_type == 1 )
	{
		// concrete noise
	    vec2 p = fbm(uv)+2.;
	    float c = length(p);
	    col.rgb = vec3(p.y)*c/15.;
	}

	else if ( noise_type == 2 )
	{
		// FBM noise
   		float n = fbm(vec3(time * 0.2,vec2(uv)))*0.5+0.5;
		col.rgb = vec3(n);
	}

	else if ( noise_type == 3 )
	{
		// Simplex3D noise
		float n = simplex3D(vec3(time,vec2(uv)))*0.5+0.5;
		col.rgb = vec3(n);
	}
	
	else if ( noise_type == 4 )
	{
		float v =0.0;
		vec2 tuv = uv / 10.;
		uv.x = tuv.x-tuv.y;
		uv.y = tuv.x+tuv.y;
		for (float i = 0.0; i<12.0; i+=1.0)
		{
			float t = mod(time+i,12.0);
			float l = time-t;
			float e = pow(1.4 * f_detail, t);
			v+=fn_noise(uv*e+vec2(cos(l)*53.0,sin(l)*100.0))*(1.0-(t/12.0))*(t/12.0);
		}
		v-=0.5;
		col = vec4(v);
	}
	
	else if ( noise_type == 5 )
	{
		float f = 0.0;
		
		if ( v_noise_type == 0 )
		{
			f = v_noise( uv * 4.);
			f = 0.5 + 0.5*f;
		}
		
		else if ( v_noise_type == 1 )
		{
			uv *= 3.0;
	        mat2 m = mat2( 1.6,  1.2, -1.2,  1.6 );
			f  = 0.5000*v_noise( uv ); uv = m*uv;
			f += 0.2500*v_noise( uv ); uv = m*uv;
			f += 0.1250*v_noise( uv ); uv = m*uv;
			f += 0.0625*v_noise( uv ); uv = m*uv;
			f = 0.5 + 0.5*f;
		}
		col.rgb = vec3(f);
		
	}
	
	else if ( noise_type == 6 )
	{
		float f = 0.0;
		f = g_noise( uv * g1_detail );
		f = 0.5 + 1.0*f;
		col.rgb = vec3(f);
	}
	
	else if ( noise_type == 7 )
	{
		float f = 0.0;
		uv *= 2.0;
        mat2 m = mat2( 1.6,  1.2, -1.2,  1.6 );
		f  = 0.5000*g_noise( uv * g2_detail ); uv = m*uv;
		f += 0.2500*g_noise( uv * g2_detail ); uv = m*uv;
		f += 0.1250*g_noise( uv * g2_detail ); uv = m*uv;
		f += 0.0625*g_noise( uv * g2_detail ); uv = m*uv;
		f = 0.5 + f;
		col.rgb = vec3(f);
	}
	
	else if ( noise_type == 8 )
	{
		float t = w_fworley(uv * resolution.xy / 1500.0);
		col.rgb = vec3(t);
	}
	
	else if ( noise_type == 9 )
	{
		uv *= scale;
		vec2 rz;
		mat2 m2 = makem2(tau/(6.+3.));
		float z=2.;
		
		if ( r_noise_type == 0 )
		{
			//base fbm noise
			for (float i= 1.;i < r_iter;i++ )
			{
				rz+= noise(uv)/z;
				z = z*.7;
				uv = uv*m2*r_detail;
			}
		}
		
		else if ( r_noise_type == 1 )
		{
			//sinus+fbm noise
			for (float i= 1.;i < r_iter;i++ )
			{
				rz+= (sin(noise(uv)*7.)*0.5+0.5) /z;
				z = z*2.;
				uv = uv*m2*r_detail;
			}	
		}
		else if ( r_noise_type == 2 )
		{
			//ridged/turbulent noise (triangle wave + fbm)
			for (float i= 1.;i < r_iter;i++ )
			{
				rz+= abs((noise(uv)-0.5)*2.)/z;
				z *= 7.;
				uv = uv*r_detail*m2;
			}
		}
		else if ( r_noise_type == 3 )
		{
			//high frenquency sinus
			for (float i= 1.;i < r_iter;i++ )
			{
				rz+= (sin(noise(uv)*25.)*0.5+0.5) /z;
				z = z*2.;
				uv = uv*m2*r_detail;
			}
		}
		col.rgb = vec3(rz.x);
	}
	
	else if ( noise_type == 10 )
	{
		float r;
	    r = fbm5(uv* 0.3);
	    r = 4.5*r-1.;
		col.rgb = clamp(vec3(r*r),0.,1.);
	}
	
	else if ( noise_type == 11 )
	{
		uv *= 200.0;
		col.rgb = vec3(p2_plasma(uv));
	}
	
	else if ( noise_type == 12 )
	{
		uv *= 200.0;
		col.rgb = vec3(p2_marble(uv / 2.0));
	}
	
	else if ( noise_type == 13 )
	{
		uv *= 200.0;
		col.rgb = vec3(p2_wood(uv));
	}
	
	else if ( noise_type == 14 )
	{
		uv *= 200.0;
		col.rgb = vec3(p2_clouds(uv));
	}
	
	else if ( noise_type == 15 )
	{
		uv*= -.2;
		uv.y += 17.0;
		float rz = crawling_flow(uv);
		col.rgb = vec3(rz*.68);
	}
		
	
	gl_FragColor = col;
}
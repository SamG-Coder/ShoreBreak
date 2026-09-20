// Shared motion for the unresolved air volume and the individually visible
// bubbles. Metres/seconds; impact time and scale come from the breaking lip.
export const ENTRAINED_AIR = /* glsl */`
struct AirPocket { float depth; float z; float ry; float rz; float life; float finger; };
AirPocket airPocket(float x,float age,float zI,float rs,float seed){
  float a=max(age,0.);
  float broad=vnoise(vec2(x*.83+seed*31.,seed*17.));
  float small=vnoise(vec2(x*2.65-seed*13.,seed*41.));
  float finger=smoothstep(.24,.77,broad*.72+small*.28);
  float plunge=min((.19+.67*finger)*rs,max(-bedHeight(vec2(x,zI))*.82,.04));
  float roll=1.-exp(-a*3.8);
  AirPocket p;
  p.depth=.035+plunge*roll*exp(-a*.40)-max(a-1.1,0.)*(.055+.035*finger);
  p.z=zI+rs*(.08+.52*(1.-exp(-a*1.1)))+.035*a;
  p.ry=.035+plunge*.43*roll;
  p.rz=.035+rs*(.14+.19*finger)*roll+.035*a;
  p.life=smoothstep(0.,.10,age)*exp(-a*.60)*(1.-smoothstep(3.,4.8,a));
  p.finger=finger;
  return p;
}
`;

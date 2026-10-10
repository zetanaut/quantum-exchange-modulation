/* Analytic controls and resource arithmetic; no quantum simulation in the browser. */
document.addEventListener("DOMContentLoaded", () => {
  const controls = document.getElementById("exchange-controls");
  if (controls) {
    const eta = controls.querySelector("[name=eta]");
    const chi = controls.querySelector("[name=chi]");
    const put = (id, value) => { controls.querySelector(`[data-value='${id}']`).textContent = value; };
    const update = () => {
      const e = Number(eta.value) * Math.PI / 180;
      const h = Number(chi.value) * Math.PI / 180;
      const a = Math.cos(e)**2, b = .5*Math.sin(2*e)*Math.cos(h);
      put("eta", `${eta.value}°`); put("chi", `${chi.value}°`);
      put("a", a.toFixed(4)); put("b", b.toFixed(4));
      put("pplus", (Math.cos(h/2)**2).toFixed(4));
      put("pminus", (Math.sin(h/2)**2).toFixed(4));
      const marker = controls.querySelector("#coefficient-marker");
      marker.setAttribute("cx", 125+200*a); marker.setAttribute("cy", 130-200*b);
    };
    eta.addEventListener("input",update); chi.addEventListener("input",update); update();
  }
  const resources = document.getElementById("resource-calculator");
  if (resources) {
    const update = () => {
      const number = name => Math.max(1,Math.floor(Number(resources.querySelector(`[name='${name}']`).value)||1));
      const n=number("qubits"), l=number("depth"), shots=number("shots"), settings=number("settings");
      const set = (name,value) => {resources.querySelector(`[data-value='${name}']`).textContent=value.toLocaleString();};
      set("copies", (l+1)*shots*settings); set("perexecution",l+1);
      set("qubits",2*n+1); set("fredkin",2*n*l); set("cnot",16*n*l);
    };
    resources.querySelectorAll("input").forEach(x=>x.addEventListener("input",update)); update();
  }
});

import React, { useState } from "react";
import Logo from "../Website_Assets/Dermalize_Logo.png";
import { BrowserRouter as Router, Route, Switch } from "react-router-dom";


function Navbar() {

  return (
    <div className="navbar">
      <div className="leftSide">
        <img src={Logo} />
        </div>
      <div className="rightSide"></div>
      </div>
  );
}

export default Navbar;
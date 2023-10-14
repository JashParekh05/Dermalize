// src/Navbar.js

import React from 'react';
import './Navbar.css'; // We'll create this CSS file for styling

function Navbar() {
  return (
    <nav className="navbar">
      <ul className="nav-list">
        <li className="nav-item"><a href="/">Home</a></li>
        <li className="nav-item"><a href="/about">About</a></li>
      </ul>
    </nav>
  );
}

export default Navbar;

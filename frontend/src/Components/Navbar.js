// src/Navbar.js

import React from 'react';
import { Link } from 'react-router-dom'; // Import Link from react-router-dom
import './Navbar.css';

function Navbar() {
  return (
    <nav className="navbar">
      <ul className="nav-list">
        <li className="nav-item"><Link to="/">Home</Link></li> {/* Use Link for navigation */}
        <li className="nav-item"><Link to="/about">About</Link></li> {/* Use Link for navigation */}
        <li className="nav-item"><a href="/services">Services</a></li>
        <li className="nav-item"><a href="/contact">Contact</a></li>
      </ul>
    </nav>
  );
}

export default Navbar;

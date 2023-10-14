import React from 'react';
import './App.css';
import Navbar from './Components/Navbar'; // Import the Navbar component

function App() {
  return (
    <div className="App">
      <Navbar /> {/* Include the Navbar component here */}
      <div className="content">
        {/* Your main content goes here */}
      </div>
    </div>
  );
}

export default App;
// src/App.js

import React from 'react';
import './App.css';
import Navbar from './Components/Navbar';
import About from './Components/AboutUs'; // Import the About component
import { BrowserRouter, Route, Switch } from 'react-router-dom';

function App() {
  return (
    <BrowserRouter>
      <div className="App">
        <Navbar />
        <div className="content">
          <Switch>
            <Route path="/about" component={About} /> {/* Define the route for "About Us" */}
            <Route path="/" exact>
              {/* Define the route for the home page */}
              <h1>Home Page</h1>
              <p>This is the home page content.</p>
            </Route>
          </Switch>
        </div>
      </div>
    </BrowserRouter>
  );
}

export default App;

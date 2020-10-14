import React, {Component} from "react";
import Toolbar from "./Toolbar";
import styles from "./ToolbarContainer.module.css";

class ToolbarContainer extends Component{
  constructor(props){
    super(props)
    this.state = {
      isSignup: true
    }
  }
  render(){
    return (
      <div className = {styles.ToolbarContainer}>
        <Toolbar/>
        <main></main>
      </div>
    )
  }
}

export default ToolbarContainer;
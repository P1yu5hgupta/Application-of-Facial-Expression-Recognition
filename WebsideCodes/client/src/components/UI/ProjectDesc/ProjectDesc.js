import React from "react"
import styles from "./ProjectDesc.module.css"

const ProjectDesc = (props)=>{
  return(
    <div className = {styles.ProjectDesc}>
      <div className = {styles.Item}>
        Emotion Tracker
      </div>
      <div>
        <ul className = {styles.Item}>
        Panel Members
          <li className = {styles.Item}>Dr. K. Vimal Kumar</li>
          <li className = {styles.Item}>Amanpreet kaur</li>
        </ul>
        <ul>
          Mentor
          <li className = {styles.Item}>Aditi Sharma</li>
        </ul>
        <ul className = {styles.Item}>
          Group
          <li className = {styles.Item}>
            Piyush Gupta(17103067)
          </li>
          <li className = {styles.Item}>
            Gaur Jetharam(17103063) 
          </li>
          <li className = {styles.Item}>
            Arpit Pundir(17103046)
          </li>
        </ul>
      </div>
    </div>
  )
}

export default ProjectDesc
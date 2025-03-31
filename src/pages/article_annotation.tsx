import React, {useState} from 'react';
import { useRouter } from 'next/router';
import { GetServerSideProps } from "next";

export const getServerSideProps: GetServerSideProps = async (context) => {
  const { email } = context.query;
  const { article } = context.query;
  if (!article || !email ) return { props: { jsonData: null, email:null, article:null, selections:null } };
  const res = await fetch(`http://localhost:3000/api/loadOneArticle/?article=${article}`); // TODO: change on deployment
  const data = await res.json();
  if (!res.ok) {
    return { props: { jsonData: null, email:null }};
  } 
  const jsonData = data.message;
  const user = String(email).split("@")[0]; // not sure if this is correct 
  let selections = null; 
  const res_selections = await fetch(`http://localhost:3000//api/loadSelections/?article=${article}&user=${user}`); // TODO: change on deployment
  const data_selections = await res_selections.json();
  if (res_selections.ok) {
    selections = data_selections.message.selections;
    console.log("GOT THE SELECTIONS: ", selections);
  }

  return { props: { jsonData, email, article, selections }};
};

export default function ArticleAnnotation({jsonData, email, article, selections}) {
  const router = useRouter();

  // State to track which paragraphs are expanded
  const [expanded, setExpanded] = useState(Array(jsonData.length).fill(false));
  const [checkboxes, setCheckboxes] = useState( () => {
    return selections ? selections : jsonData.map(() => Array(16).fill(false));
  });
  
  // TODO: change these descriptors if needed
  const frameGroups = [
    {
      name: "Civilian sympathy-mobilizing",
      frames: ["Vivid description of brutality", "Complex personhood - social and community relationships", "Complex personhood - lasting physical or psychological harm"]
    },
    {
      name: "Civilian sympathy-disrupting",
      frames: ["Prior deliquency", "Deliquency during incident"]
    },
    {
      name: "Police sympathy-mobilizing",
      frames: ["General danger or difficulty of policing", "Danger or difficulty of policing in this specific case", "Officer injury", "Context of increasing crime"]
    },
    {
      name: "Police sympathy-disrupting",
      frames: ["Previous criminal/civil lawsuits against officer or department", "Connects to previous incidents", "Highlights systematic abuses of power towards marginalized groups", "Police misconduct and lack (or slow-pace) of justice for civilian"]
    }
  ];

  const toggleExpand = (index: number) => {
      setExpanded((prev) => {
        const newState = [...prev];
        newState[index] = !newState[index]; // Toggle the selected index
        return newState;
      });
    };

  const handleCheckboxChange = (index, checkboxIndex) => {
      setCheckboxes((prev) => {
          const newCheckboxes = prev.map((checkboxGroup, i) =>
          i === index
              ? checkboxGroup.map((checked, j) =>
                  j === checkboxIndex ? !checked : checked
              )
              : checkboxGroup
          );
          return newCheckboxes;
      });
  };

  function goHome() {
    router.push({
      pathname: '/dashboard',
      query: { email: email, 
      },
    })}

  const saveSelectionsToServer = async () => {
      try {
        const response = await fetch("/api/save-selections", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email: email, article: article, selections: checkboxes }),
        });
  
        if (response.ok) {
          alert("Selections saved successfully!");
        } else {
          alert("Failed to save selections.");
        }
      } catch (error) {
        console.error("Error saving selections:", error);
      } finally {
        goHome();
      }
    };
  
  // TODO: read all the article paths from data/articles/*.json
  // and display them as links in the dashboard.

  // const article_fnames = fs.readdirSync('data/articles');
  // const names = article_fnames.map((fname: string) => {
  //     return fname.split('_')[0];
  // });
  
  return (
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          alignItems: "center",
          minHeight: "100vh",
          textAlign: "center",
          padding: "50px",
        }}
      >
        <h1>Welcome to the Dashboard!</h1>
  
        {/* Paragraphs Section */}
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", width: "100%" }}>
          {jsonData.map((paragraph, index) => (
            <div
              key={index}
              style={{
                maxWidth: "600px",
                width: "100%",
                textAlign: "left",
                marginBottom: "25px",
                padding: "15px",
                border: "1px solid #ddd",
                borderRadius: "8px",
                backgroundColor: "#f9f9f9",
              }}
            >
              {/* Paragraph with Expand Button */}
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                <p style={{ flex: 1, lineHeight: "1.8", marginRight: "10px" }}>{paragraph}</p>
                <button
                  onClick={() => toggleExpand(index)}
                  style={{
                    width: "30px",
                    height: "30px",
                    borderRadius: "5px",
                    border: "none",
                    backgroundColor: "#007bff",
                    color: "white",
                    cursor: "pointer",
                    fontSize: "18px",
                    fontWeight: "bold",
                  }}
                >
                  {expanded[index] ? "−" : "+"}
                </button>
              </div>
  
              {/* Collapsible Menu (Now Below the Paragraph) */}
              {expanded[index] && (
                <div
                  style={{
                    marginTop: "10px",
                    padding: "10px",
                    border: "1px solid #ddd",
                    borderRadius: "5px",
                    backgroundColor: "#ffffff",
                  }}
                >
                  {frameGroups.map((group, groupIndex) => (
                  <div key={groupIndex} style={{ marginBottom: "15px" }}>
                    <h4 style={{ marginBottom: "8px", marginTop: "5px", fontWeight: "600" }}>{group.name}</h4>
                    <div style={{ display: "flex", flexWrap: "wrap", flexDirection: "column"}}>
                      {group.frames.map((frameId, frameIndex) => {
                        const globalFrameIndex = groupIndex * 4 + frameIndex; // overall index
                        return (
                          <div 
                            key={frameIndex} 
                            style={{ 
                              marginRight: "15px",
                              marginBottom: "8px",
                            }}
                          >
                            <label 
                              style={{ 
                                display: "flex", 
                                alignItems: "center",
                                cursor: "pointer"
                              }}
                            >
                              <input
                                type="checkbox"
                                checked={checkboxes[index][globalFrameIndex]}
                                style={{ margin: "0 8px 0 0" }}
                                onChange={() => handleCheckboxChange(index, globalFrameIndex)}
                              />
                               <span>{frameId}</span>
                            </label>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
        {/* /* Logout Button */}
        {/* <button
          onClick={() => {
            router.push("/");
          }}
          style={{
            marginTop: "20px",
            padding: "10px 20px",
            border: "none",
            borderRadius: "5px",
            backgroundColor: "#dc3545",
            color: "white",
            fontSize: "16px",
            cursor: "pointer",
          }}
        >
          Logout
        </button> */} 
        {/* Save Button */}
      <button
          onClick={saveSelectionsToServer}
          style={{
          marginTop: "20px",
          padding: "10px 20px",
          border: "none",
          borderRadius: "5px",
          backgroundColor: "#28a745",
          color: "white",
          fontSize: "16px",
          cursor: "pointer",
          }}
      >
          Save
      </button>
      </div>
    );
};

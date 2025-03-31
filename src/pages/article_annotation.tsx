import React, {useState} from 'react';
import { useRouter } from 'next/router';
import { GetServerSideProps } from "next";

export const getServerSideProps: GetServerSideProps = async (context) => {
  const { email } = context.query; // TODO: select user-specific copy
  const { article } = context.query;
  if (!article || !email ) return { props: { jsonData: null, email:null } };
  const res = await fetch(`http://localhost:3000//api/loadOneArticle/?article=${article}`); // TODO: change on deployment
  const data = await res.json();
  if (!res.ok) {
    return { props: { jsonData: null, email:null }};
  } 
  const jsonData = data.message;
  

  return { props: { jsonData, email, article }};
};

export default function ArticleAnnotation({jsonData, email, article}) {

    // State to track which paragraphs are expanded
    const [expanded, setExpanded] = useState(Array(jsonData.length).fill(false));
    const [checkboxes, setCheckboxes] = useState(
        jsonData.map(() => [false, false, false]) // Initialize each paragraph with 3 checkboxes
      );

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
                    {checkboxes[index].map((checked, checkboxIndex) => (
                  <label key={checkboxIndex}>
                    <input
                      type="checkbox"
                      checked={checked}
                      onChange={() => handleCheckboxChange(index, checkboxIndex)}
                    />
                    Option {checkboxIndex + 1}
                  </label>
                ))}
                  </div>
                )}
              </div>
            ))}
          </div>
    
          {/* Logout Button
          <button
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

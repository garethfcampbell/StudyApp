// AI Tutor Application JavaScript

class AITutor {
    constructor() {
        console.log('AITutor constructor called');
        this.chatContainer = document.getElementById('chatContainer');
        this.chatInput = document.getElementById('chatInput');
        this.sendBtn = document.getElementById('sendBtn');
        this.loadingIndicator = document.getElementById('loadingIndicator');
        this.quizContainer = document.getElementById('quizContainer');
        this.quizContent = document.getElementById('quizContent');
        this.equationContainer = document.getElementById('equationsContainer');
        this.equationContent = document.getElementById('equationsContent');
        
        console.log('Quiz container:', this.quizContainer);
        console.log('Quiz content:', this.quizContent);
        
        this.currentQuiz = null;
        this.currentEquationPractice = null;
        
        console.log('AITutor initialized successfully');
    }
    
    showLoading() {
        if (this.loadingIndicator) {
            this.loadingIndicator.style.display = 'flex';
            this.loadingIndicator.classList.add('show');
        }
    }

    hideLoading() {
        if (this.loadingIndicator) {
            this.loadingIndicator.style.display = 'none';
            this.loadingIndicator.classList.remove('show');
        }
    }

    hideChatInterface() {
        // Hide messages container
        const messagesContainer = document.getElementById('messages');
        if (messagesContainer) {
            messagesContainer.style.display = 'none';
        }
        
        // Hide chat input section
        const chatInputSection = document.querySelector('.chat-input');
        if (chatInputSection) {
            chatInputSection.style.display = 'none';
        }
        
        // Hide revision techniques panel - using correct ID selector
        const quickActions = document.getElementById('revisionTechniques');
        if (quickActions) {
            quickActions.style.display = 'none';
            console.log('📝 DEBUG: Hidden Quick Actions panel in hideChatInterface');
        }
    }

    showChatInterface() {
        // Show messages container
        const messagesContainer = document.getElementById('messages');
        if (messagesContainer) {
            messagesContainer.style.display = 'block';
        }
        
        // Only show chat input if Executive Summary has been generated
        // Check if there are any assistant messages (indicating summary exists)
        const assistantMessages = messagesContainer ? messagesContainer.querySelectorAll('.message.assistant') : [];
        const chatInputSection = document.querySelector('.chat-input');
        if (chatInputSection && assistantMessages.length > 0) {
            chatInputSection.style.display = 'block';
        }
        
        // Only show revision techniques panel if quiz is NOT active
        const quizContainer = document.getElementById('quizContainer');
        const isQuizActive = quizContainer && quizContainer.style.display === 'block';
        
        if (!isQuizActive) {
            const quickActions = document.getElementById('revisionTechniques');
            if (quickActions) {
                quickActions.style.display = 'block';
                console.log('📝 DEBUG: Showed Quick Actions panel in showChatInterface (quiz not active)');
            }
        } else {
            console.log('📝 DEBUG: Quiz is active, keeping Quick Actions hidden in showChatInterface');
        }
        
        // Hide calculation answer input if it exists
        const calcAnswerSection = document.getElementById('calculationAnswerInput');
        if (calcAnswerSection) {
            calcAnswerSection.style.display = 'none';
            console.log('📝 DEBUG: Hidden calculation answer input when showing chat interface');
        }
        
        // Hide calculation feedback actions if they exist
        const feedbackActionsSection = document.getElementById('calculationFeedbackActions');
        if (feedbackActionsSection) {
            feedbackActionsSection.style.display = 'none';
            console.log('📝 DEBUG: Hidden calculation feedback actions when showing chat interface');
        }
    }

    // Calculation Answer Input Management
    showCalculationAnswerInput() {
        console.log('📝 DEBUG: Starting showCalculationAnswerInput');
        
        // Hide general chat input
        const chatInputSection = document.querySelector('.chat-input');
        if (chatInputSection) {
            chatInputSection.style.display = 'none';
            console.log('📝 DEBUG: Hidden chat input section');
        }
        
        // Hide Quick Actions panel during calculation questions - using correct ID selector
        const quickActionsSection = document.getElementById('revisionTechniques');
        if (quickActionsSection) {
            quickActionsSection.style.display = 'none';
            console.log('📝 DEBUG: Hidden Quick Actions panel');
        }
        
        // Find calculation answer input by ID
        let calcAnswerSection = document.getElementById('calculationAnswerInput');
        console.log('📝 DEBUG: Found calc answer section:', calcAnswerSection);
        
        if (!calcAnswerSection) {
            console.log('📝 DEBUG: Calc answer section not found, creating new one');
            // If not found, create it and insert in correct position
            calcAnswerSection = document.createElement('div');
            calcAnswerSection.id = 'calculationAnswerInput';
            calcAnswerSection.className = 'calculation-answer-input';
            calcAnswerSection.innerHTML = `
                <h5><i class="fas fa-calculator me-2"></i>Your Answer (RECREATED)</h5>
                <div class="input-group">
                    <input type="text" id="calcAnswerInput" class="form-control" 
                           placeholder="Enter your numerical answer (e.g., 4.68, 15.2%, $250)..."
                           onkeypress="if(event.key==='Enter') submitCalculationAnswer()">
                    <button class="btn btn-success" id="calcSubmitBtn" onclick="submitCalculationAnswer()">
                        <i class="fas fa-check"></i> Check Answer
                    </button>
                </div>
                <div class="mt-2">
                    <button class="btn btn-outline-secondary btn-sm me-2" onclick="nextCalculationQuestion()">
                        <i class="fas fa-forward"></i> Next Question
                    </button>
                    <button class="btn btn-outline-danger btn-sm" onclick="endCalculationSession()">
                        <i class="fas fa-stop"></i> End Practice
                    </button>
                </div>
                <div id="calcAnswerLoading" style="display: none;" class="mt-2">
                    <div class="spinner-border spinner-border-sm" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <span class="ms-2">Checking your answer...</span>
                </div>
            `;
            
            // Insert at end of main content (since quick actions are now hidden)
            const mainContent = document.querySelector('.col-md-8');
            if (mainContent) {
                mainContent.appendChild(calcAnswerSection);
                console.log('📝 DEBUG: Appended calc section to main content');
            }
        }
        
        if (calcAnswerSection) {
            calcAnswerSection.style.display = 'block';
            console.log('📝 DEBUG: Made calc answer section visible');
            
            // Ensure the input form is visible for new questions
            const calcAnswerInputGroup = document.querySelector('#calculationAnswerInput .input-group');
            if (calcAnswerInputGroup) {
                calcAnswerInputGroup.style.display = 'flex';
                console.log('📝 DEBUG: Made input form visible');
            }
            
            // Focus on the input field
            const calcInput = document.getElementById('calcAnswerInput');
            if (calcInput) {
                calcInput.focus();
                console.log('📝 DEBUG: Focused on calc input');
            }
        }
        
        console.log('📝 DEBUG: Completed showCalculationAnswerInput');
    }
    
    hideCalculationAnswerInput() {
        // Hide calculation answer input
        const calcAnswerSection = document.getElementById('calculationAnswerInput');
        if (calcAnswerSection) {
            calcAnswerSection.style.display = 'none';
        }
        
        // Hide feedback actions section too
        const feedbackActionsSection = document.getElementById('calculationFeedbackActions');
        if (feedbackActionsSection) {
            feedbackActionsSection.style.display = 'none';
        }
        
        // Only show general chat input if Executive Summary has been generated
        const messagesContainer = document.getElementById('messages');
        const assistantMessages = messagesContainer ? messagesContainer.querySelectorAll('.message.assistant') : [];
        const chatInputSection = document.querySelector('.chat-input');
        if (chatInputSection && assistantMessages.length > 0) {
            chatInputSection.style.display = 'block';
        }
        
        // Show Revision Techniques panel again
        const revisionPanel = document.getElementById('revisionTechniques');
        if (revisionPanel) {
            revisionPanel.style.display = 'block';
            console.log('Revision Techniques panel shown after calculation practice');
        }
        
        // Clear the calculation input
        const calcInput = document.getElementById('calcAnswerInput');
        if (calcInput) {
            calcInput.value = '';
        }
        
        console.log('Switched back to general chat input mode');
    }
    
    async submitCalculationAnswer() {
        const calcInput = document.getElementById('calcAnswerInput');
        if (!calcInput) return;
        
        const answer = calcInput.value.trim();
        if (!answer) {
            alert('Please enter an answer before submitting.');
            return;
        }
        
        console.log('Submitting calculation answer:', answer);
        
        try {
            // Send the answer through the simple chat endpoint
            const response = await fetch('/simple_chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: answer
                })
            });
            
            const result = await parseJSONResponse(response);
            
            if (result.success) {
                // Add the answer to chat messages
                this.addMessage('user', answer);
                
                // Add the response from AI
                this.addMessage('assistant', result.response);
                
                console.log('📝 DEBUG: Completely hiding calculation answer input after feedback');
                
                // Completely hide the entire calculation answer input section
                const calcAnswerSection = document.getElementById('calculationAnswerInput');
                console.log('📝 DEBUG: Found calc answer section:', calcAnswerSection);
                if (calcAnswerSection) {
                    calcAnswerSection.style.display = 'none';
                    console.log('📝 DEBUG: Completely hidden calculation answer input section');
                } else {
                    console.log('📝 DEBUG: Could not find calculation answer input section to hide');
                }
                
                // Show the feedback actions (Next Question and End Practice buttons)
                let feedbackActionsSection = document.getElementById('calculationFeedbackActions');
                if (!feedbackActionsSection) {
                    // Create the feedback actions section if it doesn't exist
                    feedbackActionsSection = document.createElement('div');
                    feedbackActionsSection.id = 'calculationFeedbackActions';
                    feedbackActionsSection.className = 'calculation-feedback-actions mt-3';
                    feedbackActionsSection.innerHTML = `
                        <div class="d-flex gap-2">
                            <button type="button" class="btn btn-primary" onclick="aiTutor.nextCalculationQuestion()">
                                <i class="fas fa-arrow-right"></i> Next Question
                            </button>
                            <button type="button" class="btn btn-secondary" onclick="aiTutor.endCalculationPractice()">
                                <i class="fas fa-times"></i> End Practice
                            </button>
                        </div>
                    `;
                    
                    // Insert after the messages container
                    const messagesContainer = document.getElementById('messages');
                    if (messagesContainer && messagesContainer.parentNode) {
                        messagesContainer.parentNode.insertBefore(feedbackActionsSection, messagesContainer.nextSibling);
                    }
                }
                
                if (feedbackActionsSection) {
                    feedbackActionsSection.style.display = 'block';
                    console.log('📝 DEBUG: Showing feedback actions (Next Question and End Practice buttons)');
                }
                
                // Clear the input field
                calcInput.value = '';
                
                console.log('📝 DEBUG: Calculation answer processing completed successfully');
                return true; // Return success for promise handling
            } else {
                console.error('Error submitting calculation answer:', result.error);
                alert('Error checking your answer: ' + result.error);
                throw new Error(result.error); // Throw error for promise handling
            }
            
        } catch (error) {
            console.error('Error submitting calculation answer:', error);
            alert('Error submitting your answer. Please try again.');
            throw error; // Re-throw for promise handling
        }
    }
    
    endCalculationPractice() {
        console.log('Calculation practice ending - showing Revision Techniques panel');
        
        // Immediately hide calculation interface and show Revision Techniques panel
        this.hideCalculationAnswerInput();
        
        // Send end command through chat for server-side cleanup
        fetch('/simple_chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: 'end practice'
            })
        })
        .then(response => parseJSONResponse(response))
        .then(result => {
            if (result.success) {
                this.addMessage('assistant', result.response);
            }
        })
        .catch(error => {
            console.error('Error ending calculation practice:', error);
            // UI is already cleaned up, just log the error
        });
    }
    
    nextCalculationQuestion() {
        console.log('Generating next calculation question');
        
        // Hide the feedback actions section
        const feedbackActionsSection = document.getElementById('calculationFeedbackActions');
        if (feedbackActionsSection) {
            feedbackActionsSection.style.display = 'none';
        }
        
        // Show the calculation answer input section again
        const calcAnswerSection = document.getElementById('calculationAnswerInput');
        if (calcAnswerSection) {
            calcAnswerSection.style.display = 'block';
            console.log('📝 DEBUG: Restored calculation answer input section for new question');
        }
        
        // Clear the current input
        const calcInput = document.getElementById('calcAnswerInput');
        if (calcInput) {
            calcInput.value = '';
        }
        
        // Generate a new calculation question using the global quickAction function
        if (typeof quickAction === 'function') {
            quickAction('Calculation questions');
        } else {
            console.error('quickAction function not found');
        }
    }
    
    setupDragAndDrop() {
        const uploadArea = this.uploadForm;
        
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, this.preventDefaults, false);
        });
        
        ['dragenter', 'dragover'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => uploadArea.classList.add('dragover'), false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => uploadArea.classList.remove('dragover'), false);
        });
        
        uploadArea.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.fileInput.files = files;
                this.uploadForm.dispatchEvent(new Event('submit'));
            }
        }, false);
    }
    
    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    async parseJSONResponse(response) {
        const contentType = response.headers.get('content-type');
        const responseText = await response.text();
        
        // Check if response is HTML (common when OpenAI returns error pages)
        if (contentType && contentType.includes('text/html') || responseText.trim().startsWith('<')) {
            console.error('Received HTML response instead of JSON:', responseText.substring(0, 200));
            throw new Error('Service temporarily unavailable. Please try again in a few moments.');
        }
        
        try {
            return JSON.parse(responseText);
        } catch (error) {
            console.error('JSON parsing error:', error);
            console.error('Response text:', responseText.substring(0, 200));
            // If it's an HTML response that wasn't caught above
            if (responseText.includes('<html>') || responseText.includes('<!DOCTYPE')) {
                throw new Error('Service temporarily unavailable. Please try again in a few moments.');
            }
            throw new Error('Invalid response format. Please try again.');
        }
    }
    
    async loadChatMessages() {
        // DISABLED: This function was causing content reloading issues
        console.log('🔥 AITutor loadChatMessages DISABLED to prevent content reloading');
        return;
    }
    
    addMessageToDOM(role, content) {
        console.log('=== ADDMESSAGETODOM DEBUG START ===');
        console.log('Role:', role);
        console.log('Content received:', content);
        console.log('Content length:', content.length);
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${role} fade-in`;
        
        const icon = role === 'user' ? 'fas fa-user' : 'fas fa-robot';
        
        console.log('Calling formatMessage...');
        const formattedContent = this.formatMessage(content);
        console.log('formatMessage returned:', formattedContent);
        console.log('formatMessage return length:', formattedContent.length);
        
        // Create safe DOM structure using createElement and appendChild
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        const iconElement = document.createElement('i');
        iconElement.className = `${icon} me-2`;
        
        const messageText = document.createElement('div');
        messageText.className = 'message-text';
        
        // Safely parse formatted content using DOMParser to prevent XSS
        const parser = new DOMParser();
        const doc = parser.parseFromString(formattedContent, 'text/html');
        
        // Move all child nodes from parsed document body to message text element
        while (doc.body.firstChild) {
            messageText.appendChild(doc.body.firstChild);
        }
        
        messageContent.appendChild(iconElement);
        messageContent.appendChild(messageText);
        messageDiv.appendChild(messageContent);
        
        console.log('messageDiv HTML set safely with DOM methods');
        
        this.chatContainer.appendChild(messageDiv);
        console.log('Message appended to DOM');
        
        // Check for math expressions
        const hasMath = content.includes('$') || content.includes('\\(') || content.includes('\\[') || content.includes('\\]');
        console.log('Has math expressions:', hasMath);
        console.log('  Contains $:', content.includes('$'));
        console.log('  Contains \\(:', content.includes('\\('));
        console.log('  Contains \\[:', content.includes('\\['));
        console.log('  Contains \\]:', content.includes('\\]'));
        
        // Render math if present - check for LaTeX delimiters
        if (hasMath) {
            console.log('LaTeX detected, rendering math...', content.substring(0, 200));
            console.log('Starting math rendering in 200ms...');
            // Give the DOM a moment to update before rendering
            setTimeout(() => {
                console.log('Math rendering timeout triggered, calling loadAndRenderMath...');
                this.loadAndRenderMath(messageDiv);
            }, 200);
        } else {
            console.log('No math expressions detected, skipping MathJax rendering');
        }
        
        console.log('=== ADDMESSAGETODOM DEBUG END ===');
    }
    
    async sendMessage() {
        const message = this.chatInput.value.trim();
        if (!message) return;
        
        console.log('🔥 SENDMESSAGE DEBUG: Starting sendMessage with:', message);
        
        this.chatInput.value = '';
        this.addMessage('user', message);
        this.setLoading(true);
        
        this.showLoading();
        
        try {
            console.log('🔥 SENDMESSAGE DEBUG: Sending POST to /chat');
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message })
            });
            
            console.log('🔥 SENDMESSAGE DEBUG: Response status:', response.status);
            const data = await this.parseJSONResponse(response);
            console.log('🔥 SENDMESSAGE DEBUG: Response data:', data);
            
            if (response.ok) {
                // Check if we need to start async chat processing
                if (data.start_async_chat) {
                    console.log('🔥 SENDMESSAGE DEBUG: Starting async chat processing');
                    this.addMessage('assistant', data.response);
                    // Clear initial loading indicators since async processing will handle its own
                    this.setLoading(false);
                    this.hideLoading();
                    this.startChatPolling(data.user_message);
                } else if (data.start_answer_check) {
                    console.log('🔥 SENDMESSAGE DEBUG: Starting answer check');
                    // Handle calculation answer checking
                    this.addMessage('assistant', data.response);
                    // Clear initial loading indicators since async processing will handle its own
                    this.setLoading(false);
                    this.hideLoading();
                    this.pollCalculationAnswerTask(data.challenge_question, data.user_answer);
                } else {
                    console.log('🔥 SENDMESSAGE DEBUG: Regular response');
                    this.addMessage('assistant', data.response);
                }
            } else {
                console.log('🔥 SENDMESSAGE DEBUG: Error response');
                this.addMessage('assistant', `Error: ${data.error}`);
            }
        } catch (error) {
            console.error('🔥 SENDMESSAGE DEBUG: Error in sendMessage:', error);
            this.addMessage('assistant', `Error: ${error.message}`);
        } finally {
            this.setLoading(false);
            this.hideLoading();
        }
    }

    async startChatPolling(userMessage) {
        try {
            console.log('🔥 STARTCHATPOLLING DEBUG: Starting with message:', userMessage);
            
            // Start async chat response generation
            const startResponse = await fetch('/start_chat_response', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: userMessage })
            });

            console.log('🔥 STARTCHATPOLLING DEBUG: Start response status:', startResponse.status);

            if (!startResponse.ok) {
                const errorData = await this.parseJSONResponse(startResponse);
                console.log('🔥 STARTCHATPOLLING DEBUG: Error response:', errorData);
                this.addMessage('assistant', `Error starting chat response: ${errorData.error || 'Unknown error'}`);
                return;
            }

            const startData = await this.parseJSONResponse(startResponse);
            console.log('🔥 STARTCHATPOLLING DEBUG: Start data:', startData);
            const taskId = startData.task_id;
            console.log('🔥 STARTCHATPOLLING DEBUG: Task ID:', taskId);

            // Show progress indicator
            console.log('🔥 STARTCHATPOLLING DEBUG: Showing in-chat loading');
            this.showInChatLoading();

            // Poll for completion
            console.log('🔥 STARTCHATPOLLING DEBUG: Starting pollChatResponseTask');
            this.pollChatResponseTask(taskId);

        } catch (error) {
            console.error('🔥 STARTCHATPOLLING DEBUG: Error in startChatPolling:', error);
            this.addMessage('assistant', `Error: ${error.message}`);
        }
    }

    async pollChatResponseTask(taskId) {
        console.log('🔥 POLLCHATRESPONSE DEBUG: Starting polling for task:', taskId);
        const maxAttempts = 45; // 90 seconds with 2-second intervals
        let attempts = 0;

        const pollInterval = setInterval(async () => {
            attempts++;
            console.log(`🔥 POLLCHATRESPONSE DEBUG: Polling attempt ${attempts}/${maxAttempts} for task ${taskId}`);
            
            try {
                const statusResponse = await fetch(`/chat_response_status/${taskId}`);
                console.log('🔥 POLLCHATRESPONSE DEBUG: Status response status:', statusResponse.status);
                
                if (!statusResponse.ok) {
                    console.error('🔥 POLLCHATRESPONSE DEBUG: Error response:', statusResponse.statusText);
                    clearInterval(pollInterval);
                    this.hideInChatLoading();
                    this.addMessage('assistant', 'Error checking response status. Please try again.');
                    return;
                }

                const statusData = await this.parseJSONResponse(statusResponse);
                console.log('🔥 POLLCHATRESPONSE DEBUG: Status data:', statusData);
                console.log('🔥 POLLCHATRESPONSE DEBUG: statusData.status:', statusData.status);
                console.log('🔥 POLLCHATRESPONSE DEBUG: statusData.success:', statusData.success);
                console.log('🔥 POLLCHATRESPONSE DEBUG: typeof statusData.success:', typeof statusData.success);
                console.log('🔥 POLLCHATRESPONSE DEBUG: statusData.success === true:', statusData.success === true);
                
                if (statusData.status === 'complete' && statusData.success === true) {
                    console.log('🔥 POLLCHATRESPONSE DEBUG: Task completed successfully!');
                    clearInterval(pollInterval);
                    this.hideInChatLoading();
                    
                    // Response is already added to messages by the backend, add it to UI
                    console.log('🔥 POLLCHATRESPONSE DEBUG: Adding new response to chat...');
                    console.log('🔥🔥🔥 POLLING RESPONSE DATA DEBUG 🔥🔥🔥');
                    console.log('statusData.data type:', typeof statusData.data);
                    console.log('statusData.data length:', statusData.data ? statusData.data.length : 'null/undefined');
                    console.log('FULL statusData.data content:');
                    console.log(statusData.data);
                    console.log('🔥🔥🔥 END POLLING RESPONSE DATA 🔥🔥🔥');
                    
                    console.log('🔥 ABOUT TO CALL addMessage with role=assistant');
                    console.log('🔥 Messages container before addMessage:', document.getElementById('messages'));
                    console.log('🔥 Current message count before addMessage:', document.querySelectorAll('.message').length);
                    
                    this.addMessage('assistant', statusData.data);
                    
                    console.log('🔥 Messages container after addMessage:', document.getElementById('messages'));
                    console.log('🔥 Current message count after addMessage:', document.querySelectorAll('.message').length);
                    console.log('🔥 Assistant messages count:', document.querySelectorAll('.message.assistant').length);
                    console.log('🔥 Messages with gray background count:', document.querySelectorAll('.message.assistant').length);
                    
                } else if (statusData.status === 'failed' || !statusData.success) {
                    console.log('🔥 POLLCHATRESPONSE DEBUG: Task failed:', statusData.error);
                    clearInterval(pollInterval);
                    this.hideInChatLoading();
                    this.addMessage('assistant', `Chat response failed: ${statusData.error || 'Unknown error'}`);
                    
                } else if (attempts >= maxAttempts) {
                    console.log('🔥 POLLCHATRESPONSE DEBUG: Polling timed out');
                    clearInterval(pollInterval);
                    this.hideInChatLoading();
                    this.addMessage('assistant', 'Chat response timed out. Please try again.');
                } else {
                    console.log('🔥 POLLCHATRESPONSE DEBUG: Task still pending, status:', statusData.status);
                }
                
            } catch (error) {
                console.error('🔥 POLLCHATRESPONSE DEBUG: Error in polling:', error);
                clearInterval(pollInterval);
                this.hideInChatLoading();
                this.addMessage('assistant', `Error: ${error.message}`);
            }
        }, 2000); // Poll every 2 seconds
    }

    showInChatLoading() {
        // Add a loading message in the chat
        const loadingMessage = document.createElement('div');
        loadingMessage.className = 'message assistant-message loading-message';
        loadingMessage.innerHTML = `
            <div class="message-content">
                <div class="spinner-border spinner-border-sm text-primary" role="status" style="display: inline-block;">
                    <span class="sr-only">Loading...</span>
                </div>
                <span style="margin-left: 10px;">Generating response...</span>
            </div>
        `;
        
        const messagesContainer = document.getElementById('messages');
        if (messagesContainer) {
            messagesContainer.appendChild(loadingMessage);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }
    }

    hideInChatLoading() {
        // Remove loading message from chat
        const loadingMessage = document.querySelector('.loading-message');
        if (loadingMessage) {
            loadingMessage.remove();
        }
    }
    
    addMessage(role, content) {
        // Use the global addMessage function from templates
        if (typeof window.addMessage === 'function') {
            window.addMessage(role, content);
        } else {
            // Fallback if global function not available
            console.log(`${role}: ${content}`);
        }
    }

    async loadMessages() {
        // DISABLED: This function was causing complete content reloading after chat responses
        console.log('🔥 LOADMESSAGES DEBUG: Function disabled to prevent content reloading');
        return;
        
        console.log('🔥 LOADMESSAGES DEBUG: Loading messages from backend');
        try {
            const response = await fetch('/get_messages');
            if (!response.ok) {
                console.error('🔥 LOADMESSAGES DEBUG: Error response:', response.statusText);
                return;
            }
            
            const data = await this.parseJSONResponse(response);
            console.log('🔥 LOADMESSAGES DEBUG: Received messages:', data);
            
            if (data.messages && Array.isArray(data.messages)) {
                const messagesContainer = document.getElementById('messages');
                if (messagesContainer) {
                    // Get existing message count to avoid duplication
                    const existingMessages = messagesContainer.querySelectorAll('.message').length;
                    
                    // Only add new messages that aren't already displayed
                    const newMessages = data.messages.slice(existingMessages);
                    
                    console.log('🔥🔥🔥 AITUTOR LOADMESSAGES DEBUG 🔥🔥🔥');
                    console.log('Total messages from backend:', data.messages.length);
                    console.log('Existing messages in UI:', existingMessages);
                    console.log('New messages to add:', newMessages.length);
                    console.log('All messages from backend:');
                    data.messages.forEach((msg, index) => {
                        console.log(`AITutor Message ${index}: Role=${msg.role}, Content length=${msg.content ? msg.content.length : 'null'}`);
                        console.log(`AITutor Message ${index} full content:`, msg.content);
                    });
                    console.log('🔥🔥🔥 END AITUTOR LOADMESSAGES DEBUG 🔥🔥🔥');
                    
                    newMessages.forEach(msg => {
                        console.log(`🔥 AITutor adding new message: Role=${msg.role}, Length=${msg.content ? msg.content.length : 'null'}`);
                        this.addMessage(msg.role, msg.content);
                    });
                    
                    console.log('🔥 LOADMESSAGES DEBUG: Added', newMessages.length, 'new messages (total:', data.messages.length, ')');
                } else {
                    console.log('🔥 LOADMESSAGES DEBUG: Messages container not found');
                }
            } else {
                console.error('🔥 LOADMESSAGES DEBUG: Invalid response structure:', data);
            }
        } catch (error) {
            console.error('🔥 LOADMESSAGES DEBUG: Error loading messages:', error);
        }
    }
    
    async loadAndRenderMath(element) {
        console.log('loadAndRenderMath called for element:', element);
        
        // Wait for MathJax to be fully loaded
        let attempts = 0;
        while (!window.MathJax && attempts < 100) {
            await new Promise(resolve => setTimeout(resolve, 100));
            attempts++;
        }
        
        if (!window.MathJax) {
            console.error('MathJax failed to load after waiting');
            return;
        }
        
        // Render math
        if (window.MathJax && window.MathJax.typesetPromise) {
            try {
                console.log('Rendering math with MathJax...');
                // Clear any existing math processing
                if (window.MathJax.typesetClear) {
                    window.MathJax.typesetClear([element]);
                }
                // Force a re-render of the element
                await MathJax.typesetPromise([element]);
                console.log('Math rendering completed');
            } catch (err) {
                console.error('MathJax error:', err);
                // Try a fallback approach
                try {
                    console.log('Trying fallback MathJax rendering...');
                    await MathJax.typesetPromise();
                } catch (fallbackErr) {
                    console.error('Fallback MathJax error:', fallbackErr);
                }
            }
        } else {
            console.error('MathJax.typesetPromise not available');
        }
    }
    
    // Helper function to escape HTML entities for XSS protection
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    formatMessage(content) {
        console.log('=== FORMATMESSAGE DEBUG START ===');
        console.log('Raw input content:', content);
        console.log('Input length:', content.length);
        console.log('Contains \\\\[:', content.includes('\\\\['));
        console.log('Contains \\\\(:', content.includes('\\\\('));
        console.log('Contains underscores:', content.includes('_'));
        
        // STEP 0: ESCAPE HTML - Protect against XSS while preserving LaTeX
        // First, escape the entire content to neutralize any HTML
        content = this.escapeHtml(content);
        console.log('STEP 0 COMPLETE: HTML escaped for XSS protection');
        
        // STEP 1: ISOLATE & PROTECT - Extract all LaTeX expressions for safe storage
        const latexBlocks = [];
        let blockIndex = 0;
        let originalContent = content;
        
        // Protect LaTeX environments (align*, matrix, etc.)
        content = content.replace(/\\begin\{[^}]+\}[\s\S]*?\\end\{[^}]+\}/g, (match) => {
            console.log('STEP 1A: Protected LaTeX environment:', match);
            latexBlocks.push(match);
            return `__LATEX_BLOCK_${blockIndex++}__`;
        });
        
        // Protect display math blocks \[ ... \]
        content = content.replace(/\\\[[\s\S]*?\\\]/g, (match) => {
            console.log('STEP 1B: Protected display math:', match);
            latexBlocks.push(match);
            return `__LATEX_BLOCK_${blockIndex++}__`;
        });
        
        // Protect inline math blocks \( ... \)
        content = content.replace(/\\\([\s\S]*?\\\)/g, (match) => {
            console.log('STEP 1C: Protected inline math:', match);
            latexBlocks.push(match);
            return `__LATEX_BLOCK_${blockIndex++}__`;
        });
        
        // Protect dollar sign math (legacy support)
        content = content.replace(/\$\$[\s\S]*?\$\$/g, (match) => {
            console.log('STEP 1D: Protected display dollar math:', match);
            latexBlocks.push(match);
            return `__LATEX_BLOCK_${blockIndex++}__`;
        });
        
        content = content.replace(/\$[^$\n]+\$/g, (match) => {
            console.log('STEP 1E: Protected inline dollar math:', match);
            latexBlocks.push(match);
            return `__LATEX_BLOCK_${blockIndex++}__`;
        });
        
        // Protect individual LaTeX commands with subscripts/superscripts
        content = content.replace(/\\[a-zA-Z]+(?:\{[^}]*\})*(?:[_^]\{[^}]*\})+/g, (match) => {
            console.log('STEP 1F: Protected LaTeX command:', match);
            latexBlocks.push(match);
            return `__LATEX_BLOCK_${blockIndex++}__`;
        });
        
        console.log('STEP 1 COMPLETE: LaTeX blocks isolated:', latexBlocks.length);
        console.log('Content after isolation:', content);
        console.log('All isolated blocks:', latexBlocks);
        
        // STEP 2: PROCESS MARKDOWN - Apply markdown formatting to non-LaTeX content
        let beforeMarkdown = content;
        content = content.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        content = content.replace(/\*((?!\*)[^*]+)\*/g, '<em>$1</em>');  // Avoid double asterisks
        content = content.replace(/`([^`]+)`/g, '<code>$1</code>');
        
        // Handle bullet points before converting line breaks
        content = content.replace(/^[-*+]\s+(.+)$/gm, '<li>$1</li>');
        content = content.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');
        
        // Handle line breaks (LaTeX blocks are safely isolated)
        content = content.replace(/\n/g, '<br>');
        
        console.log('STEP 2 COMPLETE: Markdown processed');
        console.log('Before markdown:', beforeMarkdown.substring(0, 200));
        console.log('After markdown:', content.substring(0, 200));
        
        // STEP 3: RESTORE - Put pristine LaTeX back into the processed HTML
        console.log('STEP 3: Starting restoration...');
        for (let i = 0; i < latexBlocks.length; i++) {
            let originalLatexBlock = latexBlocks[i];
            let latexBlock = latexBlocks[i];
            
            console.log(`STEP 3.${i}: Restoring block ${i}`);
            console.log(`  Original: ${originalLatexBlock}`);
            
            // Convert double-escaped backslashes back to single backslashes for MathJax
            latexBlock = latexBlock.replace(/\\\\/g, '\\');
            console.log(`  After delimiter fix: ${latexBlock}`);
            
            content = content.replace(`__LATEX_BLOCK_${i}__`, latexBlock);
            console.log(`  Content after replacement: ${content.substring(0, 300)}`);
        }
        
        console.log('STEP 3 COMPLETE: LaTeX blocks restored');
        console.log('Final output content:', content);
        console.log('Final output length:', content.length);
        console.log('Contains math delimiters after restoration:');
        console.log('  \\[:', content.includes('\\['));
        console.log('  \\(:', content.includes('\\('));
        console.log('  $:', content.includes('$'));
        console.log('=== FORMATMESSAGE DEBUG END ===');
        
        // STEP 4: PROCESS LATEX - This happens automatically when MathJax runs
        return content;
    }
    
    // Auto-scroll removed - users start at top of messages
    
    setLoading(loading) {
        this.sendBtn.disabled = loading;
        this.chatInput.disabled = loading;
        
        if (loading) {
            this.sendBtn.innerHTML = '<div class="spinner-border spinner-border-sm" role="status"></div>';
        } else {
            this.sendBtn.innerHTML = '<i class="fas fa-paper-plane"></i>';
        }
    }
    
    async handleFileUpload(e) {
        e.preventDefault();
        
        const file = this.fileInput.files[0];
        if (!file) {
            this.showAlert('Please select a file to upload.', 'danger');
            return;
        }
        
        const formData = new FormData();
        formData.append('file', file);
        
        this.showUploadProgress(true);
        
        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            const data = await this.parseJSONResponse(response);
            
            if (response.ok) {
                this.showAlert(data.message, 'success');
                this.enableLearningTools();
                
                // Refresh the page to show new content
                setTimeout(() => {
                    location.reload();
                }, 1000);
            } else {
                this.showAlert(data.error, 'danger');
            }
        } catch (error) {
            console.error('Upload error:', error);
            this.showAlert(`Upload failed: ${error.message}`, 'danger');
        } finally {
            this.showUploadProgress(false);
        }
    }
    
    showUploadProgress(show) {
        this.uploadProgress.style.display = show ? 'block' : 'none';
        this.uploadForm.style.display = show ? 'none' : 'block';
    }
    
    showAlert(message, type) {
        // Create elements safely to prevent XSS
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
        alertDiv.setAttribute('role', 'alert');
        
        // Safely set text content (prevents XSS)
        alertDiv.textContent = message;
        
        // Create and append close button
        const closeButton = document.createElement('button');
        closeButton.type = 'button';
        closeButton.className = 'btn-close';
        closeButton.setAttribute('data-bs-dismiss', 'alert');
        alertDiv.appendChild(closeButton);
        
        // Replace content safely
        this.uploadResult.innerHTML = '';
        this.uploadResult.appendChild(alertDiv);
    }
    
    enableLearningTools() {
        document.getElementById('startQuizBtn').disabled = false;
        document.getElementById('startEquationsBtn').disabled = false;
        document.getElementById('chatInput').disabled = false;
        document.getElementById('sendBtn').disabled = false;
    }
    
    async startQuiz() {
        console.log('startQuiz called - using async polling pattern');
        
        // Show loading with styled progress bar
        const messagesDiv = document.getElementById('messages');
        const progressDiv = document.createElement('div');
        progressDiv.className = 'message assistant';
        progressDiv.innerHTML = `
            <div class="message-content">
                <div class="d-flex align-items-center">
                    <div class="me-3">
                        <i class="fas fa-question-circle fa-2x text-primary"></i>
                    </div>
                    <div class="flex-grow-1">
                        <h6 class="mb-2">🧠 Generating Quiz...</h6>
                        <div class="progress">
                            <div class="progress-bar progress-bar-striped progress-bar-animated" 
                                 role="progressbar" style="width: 0%; background-color: #FF6600;" 
                                 aria-valuenow="0" aria-valuemin="0" aria-valuemax="100">
                                <span class="visually-hidden">Creating quiz questions...</span>
                            </div>
                        </div>
                        <small class="text-muted mt-1">
                            <i class="fas fa-brain me-1"></i>
                            Creating personalized quiz questions...
                        </small>
                    </div>
                </div>
            </div>
        `;
        messagesDiv.appendChild(progressDiv);
        
        // Animate progress bar
        const progressBar = progressDiv.querySelector('.progress-bar');
        let width = 0;
        const interval = setInterval(() => {
            width += 2;
            progressBar.style.width = width + '%';
            if (width >= 100) {
                clearInterval(interval);
                progressBar.style.backgroundColor = '#D6000D';
            }
        }, 1800); // 90 seconds total animation (50 steps * 1800ms)
        
        try {
            // Start the async quiz generation
            const response = await fetch('/start_quiz_generation', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            
            const data = await this.parseJSONResponse(response);
            
            if (response.ok && data.task_id) {
                console.log('✓ Quiz generation task started:', data.task_id);
                
                // Start polling for results
                this.startQuizPolling(data.task_id, progressDiv, interval);
            } else {
                console.error('Quiz generation start failed:', data.error);
                
                // Stop progress bar
                clearInterval(interval);
                
                // Remove progress bar
                if (messagesDiv.contains(progressDiv)) {
                    messagesDiv.removeChild(progressDiv);
                }
                
                // Restore Quick Actions panel on error
                const quickActionsSection = document.getElementById('revisionTechniques');
                if (quickActionsSection) {
                    quickActionsSection.style.display = 'block';
                    console.log('📝 DEBUG: Restored Quick Actions panel after quiz error');
                }
                
                this.addMessage('assistant', '❌ Quiz generation failed: ' + (data.error || 'Unknown error'));
            }
        } catch (error) {
            console.error('Quiz generation start error:', error);
            
            // Stop progress bar
            clearInterval(interval);
            
            // Remove progress bar if still there
            if (messagesDiv.contains(progressDiv)) {
                messagesDiv.removeChild(progressDiv);
            }
            
            // Restore Quick Actions panel on error
            const quickActionsSection = document.getElementById('revisionTechniques');
            if (quickActionsSection) {
                quickActionsSection.style.display = 'block';
                console.log('📝 DEBUG: Restored Quick Actions panel after quiz error');
            }
            
            this.addMessage('assistant', '❌ Quiz generation failed: ' + error.message);
        }
    }
    
    startQuizPolling(taskId, progressDiv, progressInterval) {
        console.log('🧠 Starting quiz polling for task:', taskId);
        
        const pollInterval = 2000; // 2 seconds between polls
        const maxAttempts = 45; // Maximum 45 attempts (90 seconds)
        let attempts = 0;
        
        const poll = () => {
            attempts++;
            console.log(`🧠 Quiz polling attempt ${attempts}/${maxAttempts} for task ${taskId}`);
            
            fetch(`/quiz_status/${taskId}`)
            .then(response => response.json())
            .then(data => {
                console.log('🧠 Quiz poll response:', data);
                
                if (data.status === 'complete' && data.success) {
                    console.log('✓ Quiz generation completed');
                    
                    // Stop progress bar
                    if (progressInterval) {
                        clearInterval(progressInterval);
                    }
                    
                    // Remove progress bar
                    const messagesDiv = document.getElementById('messages');
                    if (messagesDiv && messagesDiv.contains(progressDiv)) {
                        messagesDiv.removeChild(progressDiv);
                    }
                    
                    // Set up quiz with the generated questions
                    if (data.data && data.data.length > 0) {
                        this.currentQuiz = {
                            questions: data.data,
                            currentQuestionIndex: 0,
                            totalQuestions: data.data.length,
                            score: 0
                        };
                        console.log('📝 DEBUG: About to call showQuiz() - Quick Actions should disappear NOW');
                        this.showQuiz();
                    } else {
                        this.addMessage('assistant', '❌ No quiz questions were generated. Please try again.');
                    }
                    
                } else if (data.status === 'failed' || data.status === 'error') {
                    console.error('✗ Quiz generation failed:', data.error);
                    
                    // Stop progress bar
                    if (progressInterval) {
                        clearInterval(progressInterval);
                    }
                    
                    // Remove progress bar
                    const messagesDiv = document.getElementById('messages');
                    if (messagesDiv && messagesDiv.contains(progressDiv)) {
                        messagesDiv.removeChild(progressDiv);
                    }
                    
                    this.addMessage('assistant', '❌ ' + (data.error || 'Quiz generation failed'));
                    
                } else if (data.status === 'pending' || data.status === 'running') {
                    // Task still running, continue polling
                    if (attempts < maxAttempts) {
                        setTimeout(poll, pollInterval);
                    } else {
                        console.error('✗ Quiz polling timeout');
                        
                        // Stop progress bar
                        if (progressInterval) {
                            clearInterval(progressInterval);
                        }
                        
                        // Remove progress bar
                        const messagesDiv = document.getElementById('messages');
                        if (messagesDiv && messagesDiv.contains(progressDiv)) {
                            messagesDiv.removeChild(progressDiv);
                        }
                        
                        this.addMessage('assistant', '❌ Quiz generation is taking longer than expected. Please try again.');
                    }
                } else {
                    console.error('✗ Unknown quiz task status:', data.status);
                    
                    // Stop progress bar
                    if (progressInterval) {
                        clearInterval(progressInterval);
                    }
                    
                    // Remove progress bar
                    const messagesDiv = document.getElementById('messages');
                    if (messagesDiv && messagesDiv.contains(progressDiv)) {
                        messagesDiv.removeChild(progressDiv);
                    }
                    
                    this.addMessage('assistant', '❌ Unknown error occurred during quiz generation');
                }
            })
            .catch(error => {
                console.error('✗ Quiz polling error:', error);
                
                if (attempts < maxAttempts) {
                    // Retry on network error
                    setTimeout(poll, pollInterval);
                } else {
                    // Stop progress bar
                    if (progressInterval) {
                        clearInterval(progressInterval);
                    }
                    
                    // Remove progress bar
                    const messagesDiv = document.getElementById('messages');
                    if (messagesDiv && messagesDiv.contains(progressDiv)) {
                        messagesDiv.removeChild(progressDiv);
                    }
                    
                    this.addMessage('assistant', '❌ Network error during quiz generation. Please try again.');
                }
            });
        };
        
        // Start polling immediately
        poll();
    }
    
    showQuiz() {
        console.log('Quiz appearing - hiding Revision Techniques panel');
        
        // Show quiz container
        if (this.quizContainer) {
            this.quizContainer.style.display = 'block';
        }
        
        // Hide chat input during quiz
        const chatInput = document.querySelector('.chat-input');
        if (chatInput) {
            chatInput.style.display = 'none';
        }
        
        // Hide Revision Techniques panel during quiz
        const revisionPanel = document.getElementById('revisionTechniques');
        if (revisionPanel) {
            revisionPanel.style.display = 'none';
            console.log('Revision Techniques panel hidden for quiz');
        }
        
        this.displayQuizQuestion();
    }
    
    displayQuizQuestion() {
        console.log('displayQuizQuestion called with async data');
        console.log('Current quiz:', this.currentQuiz);
        console.log('Quiz content element:', this.quizContent);
        
        // Get current question from the questions array
        const question = this.currentQuiz.questions[this.currentQuiz.currentQuestionIndex];
        const questionNumber = this.currentQuiz.currentQuestionIndex + 1;
        const progress = (questionNumber / this.currentQuiz.totalQuestions) * 100;
        
        console.log('Question:', question);
        console.log('Question number:', questionNumber);
        console.log('Progress:', progress);
        
        // Debug question structure
        if (!question) {
            console.error('Question is undefined or null');
            return;
        }
        
        console.log('Question.question:', question.question);
        console.log('Question.options:', question.options);
        console.log('Question object keys:', Object.keys(question));
        
        // Validate question structure
        if (!question.question) {
            console.error('Question.question is undefined');
            this.addMessage('assistant', '❌ Quiz error: Question text is missing');
            return;
        }
        
        if (!question.options || !Array.isArray(question.options)) {
            console.error('Question.options is not an array:', question.options);
            this.addMessage('assistant', '❌ Quiz error: Question options are malformed');
            return;
        }
        
        if (this.quizContent) {
            // Clear existing content
            this.quizContent.innerHTML = '';
            
            // Create safe DOM structure
            const progressIndicator = document.createElement('div');
            progressIndicator.className = 'progress-indicator';
            
            const progressText = document.createElement('div');
            progressText.className = 'progress-text d-flex justify-content-between align-items-center';
            progressText.style.padding = '0 1rem 1rem 1rem'; // Increased bottom padding for more space
            progressText.style.marginBottom = '1rem'; // Increased bottom margin for more space between question/score and content
            
            const questionSpan = document.createElement('span');
            questionSpan.textContent = `Question ${questionNumber} of ${this.currentQuiz.totalQuestions}`;
            questionSpan.style.fontWeight = 'bold';
            
            const scoreSpan = document.createElement('span');
            scoreSpan.textContent = `Score: ${this.currentQuiz.score}`;
            scoreSpan.style.fontWeight = 'bold';
            
            progressText.appendChild(questionSpan);
            progressText.appendChild(scoreSpan);
            
            const progressBar = document.createElement('div');
            progressBar.className = 'progress';
            
            // Create progress bar safely using DOM methods
            const progressBarInner = document.createElement('div');
            progressBarInner.className = 'progress-bar';
            progressBarInner.style.width = `${progress}%`;
            progressBarInner.style.backgroundColor = progress >= 100 ? '#D6000D' : '#FF6600';
            
            progressBar.appendChild(progressBarInner);
            
            progressIndicator.appendChild(progressText);
            progressIndicator.appendChild(progressBar);
            
            // Create question section
            const quizQuestion = document.createElement('div');
            quizQuestion.className = 'quiz-question';
            
            const questionHeader = document.createElement('h5');
            questionHeader.textContent = question.question; // Safe: uses textContent
            
            const quizOptions = document.createElement('div');
            quizOptions.className = 'quiz-options';
            
            // Create options safely
            question.options.forEach((option, index) => {
                const optionDiv = document.createElement('div');
                optionDiv.className = 'quiz-option';
                
                const input = document.createElement('input');
                input.type = 'radio';
                input.name = 'answer';
                input.value = option; // Note: form values need special handling if they contain user data
                input.id = `option${index}`;
                
                const label = document.createElement('label');
                label.setAttribute('for', `option${index}`);
                label.className = 'ms-2';
                label.textContent = option; // Safe: uses textContent
                
                optionDiv.appendChild(input);
                optionDiv.appendChild(label);
                quizOptions.appendChild(optionDiv);
            });
            
            quizQuestion.appendChild(questionHeader);
            quizQuestion.appendChild(quizOptions);
            
            // Create button section
            const buttonDiv = document.createElement('div');
            buttonDiv.className = 'd-flex justify-content-between';
            
            const endButton = document.createElement('button');
            endButton.className = 'btn btn-secondary';
            endButton.textContent = 'End Quiz';
            endButton.onclick = () => this.endQuiz();
            
            const submitButton = document.createElement('button');
            submitButton.className = 'btn btn-primary';
            submitButton.textContent = 'Submit Answer';
            submitButton.disabled = false; // Ensure button is enabled for new question
            submitButton.onclick = () => this.submitQuizAnswer();
            
            buttonDiv.appendChild(endButton);
            buttonDiv.appendChild(submitButton);
            
            // Append all elements
            this.quizContent.appendChild(progressIndicator);
            this.quizContent.appendChild(quizQuestion);
            this.quizContent.appendChild(buttonDiv);
            console.log('Quiz content populated with async data');
        } else {
            console.error('Quiz content element not found!');
        }
    }
    
    async submitQuizAnswer() {
        const selectedAnswer = document.querySelector('input[name="answer"]:checked');
        if (!selectedAnswer) {
            this.showAlert('Please select an answer.', 'warning');
            return;
        }
        
        // Prevent multiple submissions to the same question
        const submitButton = document.querySelector('.btn-primary');
        if (submitButton && submitButton.disabled) {
            console.log('Answer already submitted for this question');
            return;
        }
        
        // Disable submit button and all radio buttons to prevent multiple submissions
        if (submitButton && submitButton.textContent === 'Submit Answer') {
            submitButton.disabled = true;
            submitButton.textContent = 'Answer Submitted';
            submitButton.className = 'btn btn-secondary';
        }
        
        // Disable all radio buttons to prevent changing selection after submission
        const allRadioButtons = document.querySelectorAll('input[name="answer"]');
        allRadioButtons.forEach(radio => {
            radio.disabled = true;
        });
        
        // Get current question (capture it before any changes)
        const currentQuestionIndex = this.currentQuiz.currentQuestionIndex;
        const currentQuestion = this.currentQuiz.questions[currentQuestionIndex];
        const userAnswer = selectedAnswer.value;
        const correctAnswer = currentQuestion.correct_answer;
        const isCorrect = userAnswer.trim() === correctAnswer.trim();
        
        console.log('Question index:', currentQuestionIndex);
        console.log('User answer:', userAnswer);
        console.log('Correct answer:', correctAnswer);
        console.log('Is correct:', isCorrect);
        
        // Update score immediately if correct
        if (isCorrect) {
            this.currentQuiz.score++;
            
            // Update score display immediately
            const scoreSpan = document.querySelector('.progress-text span:last-child');
            if (scoreSpan) {
                scoreSpan.textContent = `Score: ${this.currentQuiz.score}`;
                console.log('Updated score display immediately to:', this.currentQuiz.score);
            }
        }
        
        // Show colored feedback box immediately
        this.showQuizFeedbackBox(isCorrect);
        
        // Show result immediately (client-side processing)
        this.showQuizResult({
            correct: isCorrect,
            explanation: currentQuestion.explanation,
            quiz_complete: currentQuestionIndex >= this.currentQuiz.totalQuestions - 1,
            final_score: this.currentQuiz.score,
            total_questions: this.currentQuiz.totalQuestions
        });
    }
    
    showQuizFeedbackBox(isCorrect) {
        // Create colored feedback box that appears immediately
        const feedbackBox = document.createElement('div');
        feedbackBox.className = 'quiz-feedback-box';
        
        // Apply styling for immediate colored feedback
        feedbackBox.style.cssText = `
            margin: 1rem 0 !important;
            padding: 0.75rem 1rem !important;
            border-radius: 8px !important;
            border: 2px solid ${isCorrect ? '#28a745' : '#dc3545'} !important;
            background-color: ${isCorrect ? '#d4edda' : '#f8d7da'} !important;
            color: ${isCorrect ? '#155724' : '#721c24'} !important;
            display: block !important;
            width: 100% !important;
            box-sizing: border-box !important;
            font-weight: bold !important;
            text-align: center !important;
            font-size: 1.1rem !important;
        `;
        
        feedbackBox.textContent = isCorrect ? 'Correct!' : 'Incorrect!';
        
        // Insert feedback box right after the options
        const optionsDiv = document.querySelector('.quiz-options');
        if (optionsDiv && optionsDiv.parentNode) {
            optionsDiv.parentNode.insertBefore(feedbackBox, optionsDiv.nextSibling);
        } else {
            // Fallback: add to quiz content
            this.quizContent.appendChild(feedbackBox);
        }
        
        console.log('Added immediate feedback box:', isCorrect ? 'Correct!' : 'Incorrect!');
    }
    
    showQuizResult(data) {
        console.log('Quiz result - correct:', data.correct);
        
        // Show detailed result message (explanation and correct answer)
        const resultDiv = document.createElement('div');
        resultDiv.className = 'quiz-result-details';
        resultDiv.style.cssText = `
            margin: 1rem 0 !important;
            padding: 1rem !important;
            border-radius: 8px !important;
            border: 1px solid #dee2e6 !important;
            background-color: #f8f9fa !important;
            color: #495057 !important;
            display: block !important;
            width: 100% !important;
            box-sizing: border-box !important;
        `;
        
        // Add correct answer display
        const answerPara = document.createElement('p');
        answerPara.className = 'quiz-answer';
        answerPara.style.marginBottom = '0.5rem';
        answerPara.style.fontWeight = 'bold';
        
        if (!data.correct) {
            answerPara.textContent = `The correct answer is: ${this.currentQuiz.questions[this.currentQuiz.currentQuestionIndex].correct_answer}`;
        } else {
            answerPara.textContent = `You selected the correct answer: ${this.currentQuiz.questions[this.currentQuiz.currentQuestionIndex].correct_answer}`;
        }
        
        resultDiv.appendChild(answerPara);
        
        // Show explanation if available
        if (data.explanation && data.explanation.trim()) {
            const explanationPara = document.createElement('p');
            explanationPara.className = 'quiz-explanation';
            explanationPara.style.marginBottom = '0';
            explanationPara.textContent = data.explanation;
            resultDiv.appendChild(explanationPara);
        }
        
        console.log('Adding result details to quiz content');
        this.quizContent.appendChild(resultDiv);
        
        // Debug styles after adding to DOM
        setTimeout(() => {
            const computedStyle = window.getComputedStyle(resultDiv);
            console.log('POST-DOM computed styles:');
            console.log('- Background:', computedStyle.backgroundColor);
            console.log('- Border:', computedStyle.border);
            console.log('- Border color:', computedStyle.borderColor);
            console.log('- Padding:', computedStyle.padding);
            console.log('- Display:', computedStyle.display);
            console.log('- Classes:', resultDiv.className);
            console.log('- Element HTML:', resultDiv.outerHTML);
        }, 100);
        
        if (data.quiz_complete) {
            const scoreDiv = document.createElement('div');
            scoreDiv.className = 'score-display';
            scoreDiv.textContent = `Quiz Complete! Final Score: ${data.final_score}/${data.total_questions}`;
            this.quizContent.appendChild(scoreDiv);
            
            const closeButton = document.createElement('button');
            closeButton.className = 'btn btn-primary';
            closeButton.textContent = 'Close Quiz';
            closeButton.onclick = () => aiTutor.endQuiz();
            this.quizContent.appendChild(closeButton);
        } else {
            // Move to next question
            this.currentQuiz.currentQuestionIndex++;
            
            // Show next question button using safe DOM methods
            const buttonContainer = document.createElement('div');
            buttonContainer.className = 'd-flex justify-content-center mt-3';
            
            const nextButton = document.createElement('button');
            nextButton.className = 'btn btn-primary';
            nextButton.textContent = 'Next Question';
            nextButton.onclick = () => aiTutor.showNextQuestion();
            
            buttonContainer.appendChild(nextButton);
            this.quizContent.appendChild(buttonContainer);
        }
    }
    
    showNextQuestion() {
        console.log('Moving to next question:', this.currentQuiz.currentQuestionIndex);
        this.displayQuizQuestion();
    }
    
    endQuiz() {
        console.log('Quiz ending - showing Revision Techniques panel');
        
        // Hide quiz container
        if (this.quizContainer) {
            this.quizContainer.style.display = 'none';
        }
        this.currentQuiz = null;
        
        // Show Revision Techniques panel again
        const revisionPanel = document.getElementById('revisionTechniques');
        if (revisionPanel) {
            revisionPanel.style.display = 'block';
            console.log('Revision Techniques panel shown after quiz');
        }
        
        // Show chat input again if executive summary exists
        const messagesContainer = document.getElementById('messages');
        const assistantMessages = messagesContainer ? messagesContainer.querySelectorAll('.message.assistant') : [];
        const chatInput = document.querySelector('.chat-input');
        if (chatInput && assistantMessages.length > 0) {
            chatInput.style.display = 'block';
        }
        
        // Add completion message
        this.addMessage('assistant', '🎯 Quiz completed! You can start another quiz or ask me any questions about your lecture notes.');
    }
    
    async startEquations() {
        console.log('=== startEquations called ===');
        
        // Show progress bar
        const calcButton = document.querySelector('[onclick="aiTutor.startEquations()"]');
        if (calcButton) {
            calcButton.classList.add('active');
            this.animateProgressBar(calcButton, 90000); // 90 seconds
        }
        
        try {
            // First, get the list of equations
            const listResponse = await fetch('/list_equations', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            
            const listData = await listResponse.json();
            
            if (listResponse.ok && listData.success) {
                // Add the equation list to the chat
                this.addMessage('assistant', listData.response);
                
                // Keep button active (green) to show success
                if (calcButton) {
                    calcButton.style.backgroundColor = '#28a745';
                    calcButton.style.borderColor = '#28a745';
                }
            } else {
                this.showAlert(listData.error || 'Error listing equations', 'danger');
                // Remove active state on error
                if (calcButton) {
                    calcButton.classList.remove('active');
                    calcButton.style.backgroundColor = '';
                    calcButton.style.borderColor = '';
                }
            }
        } catch (error) {
            console.error('Start equations error:', error);
            this.showAlert(`Error listing equations: ${error.message}`, 'danger');
            // Remove active state on error
            if (calcButton) {
                calcButton.classList.remove('active');
                calcButton.style.backgroundColor = '';
                calcButton.style.borderColor = '';
            }
        }
    }
    
    // showEquations method removed - calculations now display in main chat
    
    displayEquationProblem() {
        console.log('=== displayEquationProblem called ===');
        console.log('currentEquationPractice:', this.currentEquationPractice);
        console.log('currentIndex:', this.currentEquationPractice?.currentIndex);
        console.log('equations array:', this.currentEquationPractice?.equations);
        
        const equation = this.currentEquationPractice.equations[this.currentEquationPractice.currentIndex];
        console.log('Selected equation:', equation);
        console.log('equation.equation:', equation?.equation);
        console.log('equation.description:', equation?.description);
        
        const progress = ((this.currentEquationPractice.currentIndex + 1) / this.currentEquationPractice.equations.length) * 100;
        
        // Check if equation exists and has required properties
        if (!equation || !equation.equation) {
            console.error('Equation is missing or invalid:', equation);
            // Safe error display
            this.equationContent.innerHTML = '';
            const errorDiv = document.createElement('div');
            errorDiv.className = 'alert alert-danger';
            errorDiv.textContent = 'Error: Invalid equation data';
            this.equationContent.appendChild(errorDiv);
            return;
        }
        
        // Clear and build safe DOM structure
        this.equationContent.innerHTML = '';
        
        // Progress indicator
        const progressIndicator = document.createElement('div');
        progressIndicator.className = 'progress-indicator';
        
        const progressText = document.createElement('div');
        progressText.className = 'progress-text';
        
        const problemSpan = document.createElement('span');
        problemSpan.textContent = `Problem ${this.currentEquationPractice.currentIndex + 1} of ${this.currentEquationPractice.equations.length}`;
        
        const scoreSpan = document.createElement('span');
        scoreSpan.textContent = `Score: ${this.currentEquationPractice.score}`;
        
        progressText.appendChild(problemSpan);
        progressText.appendChild(scoreSpan);
        
        const progressBar = document.createElement('div');
        progressBar.className = 'progress';
        
        const progressBarInner = document.createElement('div');
        progressBarInner.className = 'progress-bar';
        progressBarInner.style.width = `${progress}%`;
        progressBar.appendChild(progressBarInner);
        
        progressIndicator.appendChild(progressText);
        progressIndicator.appendChild(progressBar);
        
        // Equation problem section
        const equationProblem = document.createElement('div');
        equationProblem.className = 'equation-problem';
        
        const header = document.createElement('h5');
        header.textContent = 'Solve the equation:';
        
        const equationDisplay = document.createElement('div');
        equationDisplay.className = 'equation-display';
        equationDisplay.textContent = equation.equation || 'No equation available'; // Safe: uses textContent
        
        const description = document.createElement('p');
        description.className = 'equation-description';
        description.textContent = equation.description || 'No description available'; // Safe: uses textContent
        
        equationProblem.appendChild(header);
        equationProblem.appendChild(equationDisplay);
        equationProblem.appendChild(description);
        
        // Add worked example if present
        if (equation.worked_example) {
            const workedExampleDiv = document.createElement('div');
            workedExampleDiv.className = 'worked-example';
            
            const strong = document.createElement('strong');
            strong.textContent = 'Worked Example:';
            
            const pre = document.createElement('pre');
            pre.textContent = equation.worked_example; // Safe: uses textContent
            
            workedExampleDiv.appendChild(strong);
            workedExampleDiv.appendChild(pre);
            equationProblem.appendChild(workedExampleDiv);
        }
        
        // Input section
        const inputDiv = document.createElement('div');
        inputDiv.className = 'mb-3';
        
        const label = document.createElement('label');
        label.setAttribute('for', 'equationAnswer');
        label.className = 'form-label';
        label.textContent = 'Your Answer:';
        
        const input = document.createElement('input');
        input.type = 'text';
        input.className = 'form-control equation-input';
        input.id = 'equationAnswer';
        input.placeholder = 'Enter your answer...';
        
        inputDiv.appendChild(label);
        inputDiv.appendChild(input);
        
        // Button section
        const buttonDiv = document.createElement('div');
        buttonDiv.className = 'd-flex justify-content-between';
        
        const endButton = document.createElement('button');
        endButton.className = 'btn btn-secondary';
        endButton.textContent = 'End Practice';
        endButton.onclick = () => this.endEquations();
        
        const submitButton = document.createElement('button');
        submitButton.className = 'btn btn-primary';
        submitButton.textContent = 'Submit Answer';
        submitButton.onclick = () => this.submitEquationAnswer();
        
        buttonDiv.appendChild(endButton);
        buttonDiv.appendChild(submitButton);
        
        // Append all elements
        this.equationContent.appendChild(progressIndicator);
        this.equationContent.appendChild(equationProblem);
        this.equationContent.appendChild(inputDiv);
        this.equationContent.appendChild(buttonDiv);
        
        // Focus on input
        setTimeout(() => {
            document.getElementById('equationAnswer').focus();
        }, 100);
        
        // Add enter key listener
        document.getElementById('equationAnswer').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.submitEquationAnswer();
            }
        });
        
        // Load MathJax and render equations
        setTimeout(() => {
            console.log('Attempting to render MathJax for equation:', equation.equation);
            
            if (typeof MathJax !== 'undefined' && MathJax.typesetPromise) {
                console.log('MathJax is available, rendering...');
                MathJax.typesetPromise([this.equationContent]).then(() => {
                    console.log('MathJax rendering complete');
                }).catch(error => {
                    console.error('MathJax rendering error:', error);
                });
            } else {
                console.log('MathJax not available, waiting for it to load...');
                // Wait for MathJax to load and then render
                const checkMathJax = setInterval(() => {
                    if (typeof MathJax !== 'undefined' && MathJax.typesetPromise) {
                        console.log('MathJax loaded, rendering now...');
                        clearInterval(checkMathJax);
                        MathJax.typesetPromise([this.equationContent]).then(() => {
                            console.log('MathJax rendering complete');
                        }).catch(error => {
                            console.error('MathJax rendering error:', error);
                        });
                    }
                }, 100);
                
                // Stop checking after 10 seconds
                setTimeout(() => {
                    clearInterval(checkMathJax);
                    console.log('MathJax loading timeout');
                }, 10000);
            }
        }, 500);
    }
    
    async submitEquationAnswer() {
        const answer = document.getElementById('equationAnswer').value.trim();
        if (!answer) {
            this.showAlert('Please enter an answer.', 'warning');
            return;
        }
        
        try {
            const response = await fetch('/submit_equation', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ answer: answer })
            });
            
            const data = await this.parseJSONResponse(response);
            
            if (response.ok) {
                this.showEquationResult(data);
            } else {
                this.showAlert(data.error, 'danger');
            }
        } catch (error) {
            console.error('Submit equation error:', error);
            this.showAlert(`Error submitting answer: ${error.message}`, 'danger');
        }
    }
    
    showEquationResult(data) {
        const resultClass = data.correct ? 'correct' : 'incorrect';
        const resultText = data.correct ? 'Correct!' : 'Incorrect';
        
        // Create safe result display
        const resultDiv = document.createElement('div');
        resultDiv.className = `quiz-result ${resultClass}`;
        
        const resultHeader = document.createElement('h6');
        resultHeader.textContent = resultText;
        
        const explanation = document.createElement('p');
        explanation.textContent = data.explanation; // Safe: uses textContent
        
        resultDiv.appendChild(resultHeader);
        resultDiv.appendChild(explanation);
        this.equationContent.appendChild(resultDiv);
        
        if (data.practice_complete) {
            const scoreDiv = document.createElement('div');
            scoreDiv.className = 'score-display';
            scoreDiv.textContent = `Practice Complete! Final Score: ${data.final_score}/${data.total_equations}`;
            
            const closeButton = document.createElement('button');
            closeButton.className = 'btn btn-primary';
            closeButton.textContent = 'Close Practice';
            closeButton.onclick = () => this.endEquations();
            
            this.equationContent.appendChild(scoreDiv);
            this.equationContent.appendChild(closeButton);
        } else {
            this.currentEquationPractice.currentIndex++;
            if (data.correct) {
                this.currentEquationPractice.score++;
            }
            
            setTimeout(() => {
                this.displayEquationProblem();
            }, 2000);
        }
    }
    
    endEquations() {
        this.equationContainer.style.display = 'none';
        this.currentEquationPractice = null;
    }
    
    addMessage(role, content) {
        // Use the global addMessage function from templates
        if (typeof window.addMessage === 'function') {
            window.addMessage(role, content);
        } else {
            // Fallback if global function not available
            console.log(`${role}: ${content}`);
        }
    }
    
    async clearSession() {
        if (confirm('Are you sure you want to clear the current session? This will remove all chat history and uploaded documents.')) {
            try {
                const response = await fetch('/clear_session', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    }
                });
                
                if (response.ok) {
                    location.reload();
                } else {
                    this.showAlert('Failed to clear session.', 'danger');
                }
            } catch (error) {
                console.error('Clear session error:', error);
                this.showAlert(`Error clearing session: ${error.message}`, 'danger');
            }
        }
    }
}

// Global functions for onclick handlers
function endCalculationSession() {
    if (window.aiTutor) {
        window.aiTutor.endCalculationPractice();
    }
}

function nextCalculationQuestion() {
    if (window.aiTutor) {
        window.aiTutor.nextCalculationQuestion();
    }
}

function submitCalculationAnswer() {
    console.log('📝 GLOBAL DEBUG: submitCalculationAnswer called');
    
    const calcInput = document.getElementById('calcAnswerInput');
    if (!calcInput) {
        console.log('📝 GLOBAL DEBUG: calcAnswerInput not found');
        return;
    }
    
    const answer = calcInput.value.trim();
    if (!answer) {
        alert('Please enter an answer before submitting.');
        return;
    }
    
    console.log('📝 GLOBAL DEBUG: Submitting answer through newSendMessage:', answer);
    
    // Use the new chat system to send the answer
    const chatInput = document.getElementById('newChatInput');
    if (chatInput) {
        chatInput.value = answer;
        newSendMessage();
        
        // Clear the calculation input
        calcInput.value = '';
        
        console.log('📝 GLOBAL DEBUG: Answer submitted through chat system');
    } else {
        console.log('📝 GLOBAL DEBUG: newChatInput not found, falling back to AITutor method');
        if (window.aiTutor) {
            window.aiTutor.submitCalculationAnswer();
        }
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    window.aiTutor = new AITutor();
});
